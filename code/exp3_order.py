# -*- coding: utf-8 -*-
"""
exp3_order.py — removal-order sensitivity of path discovery with real LLM observers (reviewer item 4).

Question. When several inclusion-minimal paths exist, the order in which cues are tried for removal decides
which one backward elimination returns. Experiment 3 used the dialogue order only. How much of the
between-observer difference in Table V could be an artifact of that fixed order?

Design. For each of the first N Patil scenarios (default 30; same prefix as Appendix A) and the benchmark
secret, backward elimination is run with the dialogue order (order 0), the reverse order (order 1) and K
random orders (default 10). Every judgment V_z(S) is cached per cue subset S, so within one scenario and
observer all orders see the SAME judgment function (r-vote majority, queried once per subset): path
differences between orders are then caused by the order alone, not by run-to-run response variation
(which Appendix A measures separately). Every distinct returned path is validated once with the five checks.

Reported per observer (--report-only):
  * distinct paths per scenario; scenarios with exactly one path
  * path length: mean, sd, min, max, and the mean within-scenario range
  * path stability: mean pairwise Jaccard between the paths of different orders (within scenario)
  * Jaccard to the benchmark-annotated cue set: mean over orders, mean within-scenario sd
  * 5/5 validation pass rate over orders, and the within-scenario variation (scenarios whose verdict
    depends on the order)
  * where the dialogue-order path sits: fraction of random orders returning the identical path, and the
    percentile of its benchmark Jaccard among all orders of the scenario
  * judgments actually sent to the API (after caching) vs. the uncached count
Cross-observer (--compare, reads the three progress files):
  * mean within-observer between-order Jaccard vs. mean between-observer Jaccard of the dialogue-order
    paths on the same scenarios; the latter being lower than the former is the observer-dependence claim.

Resumable: one JSON line per (scenario, order) in results/logs/order_progress_<backend>_<model>.jsonl;
the subset cache is stored next to it (order_cache_<tag>.json) so a rerun does not re-query.

  export OPENAI_API_KEY=...; export OPENROUTER_API_KEY=...        # never commit keys
  python3 exp3_order.py --backend openai     --model gpt-4o-mini-2024-07-18      --limit 30 --orders 10 --r 5 --raw
  python3 exp3_order.py --backend openrouter --model qwen/qwen3.6-27b             --limit 30 --orders 10 --r 5 --raw
  python3 exp3_order.py --backend openrouter --model google/gemini-2.5-flash-lite --limit 30 --orders 10 --r 5 --raw
  python3 exp3_order.py --backend openai --model gpt-4o-mini-2024-07-18 --report-only
  python3 exp3_order.py --compare                                   # all three observers, writes results/exp3_order_summary.json
Cost (with the cache): roughly 40-80 distinct subsets x r calls per scenario for discovery and
(4+|S|) x r calls per distinct path for validation; about 300-500 calls per scenario and observer.
"""
import argparse, glob, itertools, json, os, random, statistics as st, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import jaccard, QueryCounter
from exp3_endtoend import render, certify

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "results", "logs")


class CachedJudge:
    """V_z(S) queried once per subset; counts both logical queries and API-level judgments."""
    def __init__(self, obs, turns, z, cache):
        self.obs, self.turns, self.z, self.cache = obs, turns, z, cache
        self.logical = 0; self.sent = 0

    def __call__(self, S):
        key = ",".join(map(str, sorted(S)))
        self.logical += 1
        if key not in self.cache:
            self.cache[key] = self.obs.judge_excerpt(render(self.turns, sorted(S)), self.z)
            self.sent += 1
        return self.cache[key]


def backward_search_order(V, order):
    if V(list(order)) == 0:
        return None
    S = set(order)
    for i in order:
        if V(list(S - {i})) == 1:
            S = S - {i}
    return frozenset(S)


def tag_of(args):
    return f"{args.backend}_{(args.model or 'default').replace('/', '_')}"


def prog_path(tag):
    return os.path.join(LOGS, f"order_progress_{tag}.jsonl")


def cache_path(tag):
    return os.path.join(LOGS, f"order_cache_{tag}.json")


def summarize(rows, gts):
    by = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r)
    out = {"scenarios": len(by)}
    n_paths, lens, len_range, stab, jac_mean, jac_sd, pass_rate, pass_var = [], [], [], [], [], [], [], 0
    same_as_dialogue, dial_pct, no_path, sent, logical = [], [], 0, 0, 0
    for sid, rs in by.items():
        sent += sum(r.get("sent", 0) for r in rs); logical += sum(r.get("q", 0) for r in rs)
        withp = [r for r in rs if r.get("S")]
        if not withp:
            no_path += 1; continue
        paths = [frozenset(r["S"]) for r in withp]
        n_paths.append(len(set(paths)))
        L = [len(p) for p in paths]; lens.extend(L); len_range.append(max(L) - min(L))
        if len(paths) > 1:
            stab.append(st.mean(jaccard(a, b) for a, b in itertools.combinations(paths, 2)))
        js = [jaccard(p, gts[sid]) for p in paths]
        jac_mean.append(st.mean(js)); jac_sd.append(st.pstdev(js))
        verdicts = [1.0 if r.get("cert") else 0.0 for r in withp]
        pass_rate.append(st.mean(verdicts)); pass_var += int(0 < sum(verdicts) < len(verdicts))
        d = [r for r in withp if r["order_id"] == 0]
        if d:
            dp = frozenset(d[0]["S"]); others = [frozenset(r["S"]) for r in withp if r["order_id"] >= 2]
            if others:
                same_as_dialogue.append(st.mean(1.0 if p == dp else 0.0 for p in others))
            dj = jaccard(dp, gts[sid])
            dial_pct.append(st.mean(1.0 if j <= dj else 0.0 for j in js))
    out.update({"no_path_any_order": no_path, "api_judgments_sent": sent, "logical_judgments": logical})
    if n_paths:
        out.update({
            "distinct_paths_mean": st.mean(n_paths), "distinct_paths_max": max(n_paths),
            "one_path_scenarios": sum(1 for x in n_paths if x == 1), "scenarios_with_path": len(n_paths),
            "path_len_mean": st.mean(lens), "path_len_sd": st.pstdev(lens), "path_len_min": min(lens), "path_len_max": max(lens),
            "path_len_within_range_mean": st.mean(len_range),
            "stability_pairwise_jaccard": st.mean(stab) if stab else None,
            "jaccard_to_benchmark_mean": st.mean(jac_mean), "jaccard_within_sd_mean": st.mean(jac_sd),
            "pass_rate_over_orders": st.mean(pass_rate), "order_dependent_verdict_scenarios": pass_var,
            "dialogue_path_identical_to_random_fraction": st.mean(same_as_dialogue) if same_as_dialogue else None,
            "dialogue_path_jaccard_percentile_mean": st.mean(dial_pct) if dial_pct else None})
    return out


def print_summary(s):
    print(f"scenarios: {s['scenarios']} (no path under any order: {s['no_path_any_order']}); "
          f"API judgments sent {s['api_judgments_sent']} of {s['logical_judgments']} logical")
    if "distinct_paths_mean" not in s:
        return
    print(f"distinct paths per scenario: mean {s['distinct_paths_mean']:.2f}, max {s['distinct_paths_max']}; "
          f"exactly one path: {s['one_path_scenarios']}/{s['scenarios_with_path']}")
    print(f"path length: mean {s['path_len_mean']:.2f}, sd {s['path_len_sd']:.2f}, min {s['path_len_min']}, max {s['path_len_max']}; "
          f"mean within-scenario range {s['path_len_within_range_mean']:.2f}")
    print(f"path stability (mean pairwise Jaccard between orders): {s['stability_pairwise_jaccard']}")
    print(f"Jaccard to benchmark-annotated cues: mean {s['jaccard_to_benchmark_mean']:.3f}, mean within-scenario sd {s['jaccard_within_sd_mean']:.3f}")
    print(f"5/5 pass rate over orders: {s['pass_rate_over_orders']:.3f}; scenarios whose verdict depends on the order: {s['order_dependent_verdict_scenarios']}")
    print(f"dialogue-order path: identical to a random-order path in {s['dialogue_path_identical_to_random_fraction']:.3f} of random orders; "
          f"its benchmark Jaccard is at the {100*s['dialogue_path_jaccard_percentile_mean']:.0f}th percentile of the scenario's orders on average")


def compare(limit):
    files = sorted(glob.glob(os.path.join(LOGS, "order_progress_*.jsonl")))
    scen = load_patil()[:limit]; gts = {s.sid: set(s.gt) for s in scen}
    per = {}
    for f in files:
        tag = os.path.basename(f)[len("order_progress_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(f, encoding="utf8")]
        per[tag] = {"summary": summarize(rows, gts), "rows": rows}
        print(f"\n== {tag}"); print_summary(per[tag]["summary"])
    tags = list(per)
    cross = {}
    for a, b in itertools.combinations(tags, 2):
        da = {r["sid"]: frozenset(r["S"]) for r in per[a]["rows"] if r["order_id"] == 0 and r.get("S")}
        db = {r["sid"]: frozenset(r["S"]) for r in per[b]["rows"] if r["order_id"] == 0 and r.get("S")}
        common = sorted(set(da) & set(db))
        if common:
            cross[f"{a} vs {b}"] = {"n": len(common), "between_observer_jaccard_dialogue_order": st.mean(jaccard(da[s], db[s]) for s in common)}
    within = {t: per[t]["summary"].get("stability_pairwise_jaccard") for t in tags}
    print("\nwithin-observer between-order Jaccard:", {t: round(v, 3) if v is not None else None for t, v in within.items()})
    for k, v in cross.items():
        print(f"between-observer Jaccard (dialogue-order paths, n={v['n']}): {k}: {v['between_observer_jaccard_dialogue_order']:.3f}")
    out = os.path.join(HERE, "..", "results", "exp3_order_summary.json")
    json.dump({"per_observer": {t: per[t]["summary"] for t in tags}, "within_observer_between_order_jaccard": within, "between_observer": cross},
              open(out, "w"), indent=1)
    print("written:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--orders", type=int, default=10, help="random removal orders per scenario (dialogue order and reverse order are orders 0 and 1)")
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args.limit); return
    assert args.backend, "--backend is required"
    tag = tag_of(args); prog = prog_path(tag); cpath = cache_path(tag)
    done = {}
    if os.path.exists(prog):
        for line in open(prog, encoding="utf8"):
            r = json.loads(line); done[(r["sid"], r["order_id"])] = r
    scen = load_patil()[:args.limit]
    gts = {s.sid: set(s.gt) for s in scen}
    if args.report_only:
        print_summary(summarize(list(done.values()), gts)); return
    cache_all = json.load(open(cpath, encoding="utf8")) if os.path.exists(cpath) else {}
    backend = make_backend(args.backend, args.model)
    obs = Observer(backend, r=args.r)
    os.makedirs(LOGS, exist_ok=True)
    for s in scen:
        turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
        z = s.secret; n = len(turns)
        rng = random.Random(args.seed * 100003 + int(s.sid.split("_")[-1]))     # reproducible, independent of hash randomization
        orders = [list(range(n)), list(range(n))[::-1]] + [rng.sample(range(n), n) for _ in range(args.orders)]
        cache = cache_all.setdefault(s.sid, {"V": {}, "cert": {}})
        V = CachedJudge(obs, turns, z, cache["V"])
        for oid, order in enumerate(orders):
            if (s.sid, oid) in done:
                continue
            V.logical = V.sent = 0
            S = backward_search_order(V, order)
            rec = {"sid": s.sid, "order_id": oid, "order": order, "q": V.logical, "sent": V.sent}
            if S is not None:
                key = ",".join(map(str, sorted(S)))
                if len(S) >= 2:
                    if key not in cache["cert"]:
                        ok, cd = certify(obs, turns, S, z, detail=True)
                        cache["cert"][key] = {"cert": bool(ok), "cd": {k: bool(cd[k]) for k in ("base", "single", "MR1", "MR3", "MR4")}}
                    rec.update({"S": sorted(S), "jac": jaccard(S, s.gt), **cache["cert"][key]})
                else:
                    rec.update({"S": sorted(S), "jac": jaccard(S, s.gt), "cert": False, "single_cue": True})
            with open(prog, "a", encoding="utf8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            json.dump(cache_all, open(cpath, "w", encoding="utf8"))
            done[(s.sid, oid)] = rec
            if args.sleep: time.sleep(args.sleep)
        print(f"  {s.sid}: {len(orders)} orders done; distinct subsets judged so far {len(cache['V'])}, distinct paths {len(cache['cert'])}")
    print_summary(summarize(list(done.values()), gts))


if __name__ == "__main__":
    main()
