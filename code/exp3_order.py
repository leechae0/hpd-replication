# -*- coding: utf-8 -*-
"""
exp3_order.py — removal-order sensitivity of path discovery with a real LLM observer (reviewer item 4).

For each of the first N Patil scenarios (default 30) and a fixed benchmark secret, backward
elimination is run K times (default 10) with different random cue-removal orders (plus the
dialogue order used in the paper), and every returned path is validated with the five checks.
Reported per observer:
  * number of distinct inclusion-minimal paths found across orders
  * path length (mean, min, max)
  * mean pairwise Jaccard between the paths of different orders (path stability)
  * Jaccard to the benchmark-annotated cue set (mean, sd across orders)
  * validation pass rate (5/5) across orders
Resumable: one JSON line per (scenario, order) in results/logs/order_progress_<backend>_<model>.jsonl.

  export OPENAI_API_KEY=...        # never commit keys
  python3 exp3_order.py --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --orders 10 --r 5 --raw
  python3 exp3_order.py --backend openai --model gpt-4o-mini-2024-07-18 --report-only
Cost: about (|U|+1) x r x (K+1) judgments per scenario for discovery plus ~7 x r per validated path.
"""
import argparse, itertools, json, os, random, statistics as st, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import jaccard, QueryCounter
from exp3_endtoend import render, certify


def backward_search_order(V, order, qc):
    """Backward elimination over the cues in the given order (search.backward_search with an explicit order)."""
    qc.tick()
    if V(list(order)) == 0:
        return None
    S = set(order)
    for i in order:
        qc.tick()
        if V(list(S - {i})) == 1:
            S = S - {i}
    return frozenset(S)


def prog_path(args):
    tag = f"{args.backend}_{(args.model or 'default').replace('/', '_')}"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "logs", f"order_progress_{tag}.jsonl")


def report(rows, gts):
    by = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r)
    n_paths, lens, stab, jac_mean, jac_sd, pass_rate, no_path = [], [], [], [], [], [], 0
    for sid, rs in by.items():
        paths = [frozenset(r["S"]) for r in rs if r.get("S")]
        if not paths:
            no_path += 1; continue
        n_paths.append(len(set(paths)))
        lens.extend(len(p) for p in paths)
        if len(paths) > 1:
            stab.append(st.mean(jaccard(a, b) for a, b in itertools.combinations(paths, 2)))
        js = [jaccard(p, gts[sid]) for p in paths]
        jac_mean.append(st.mean(js)); jac_sd.append(st.pstdev(js))
        pass_rate.append(st.mean(1.0 if r.get("cert") else 0.0 for r in rs if r.get("S")))
    print(f"scenarios: {len(by)} (no path under any order: {no_path})")
    if n_paths:
        print(f"distinct paths per scenario: mean {st.mean(n_paths):.2f}, max {max(n_paths)}; "
              f"scenarios with exactly one path: {sum(1 for x in n_paths if x == 1)}/{len(n_paths)}")
        print(f"path length: mean {st.mean(lens):.2f}, min {min(lens)}, max {max(lens)}")
        print(f"path stability (mean pairwise Jaccard between orders): {st.mean(stab):.3f}" if stab else "path stability: n/a")
        print(f"Jaccard to benchmark-annotated cues: mean {st.mean(jac_mean):.3f}, mean within-scenario sd {st.mean(jac_sd):.3f}")
        print(f"5/5 validation pass rate across orders: {st.mean(pass_rate):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--orders", type=int, default=10, help="random removal orders per scenario (dialogue order is added as order 0)")
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    prog = prog_path(args)
    done = {}
    if os.path.exists(prog):
        for line in open(prog, encoding="utf8"):
            r = json.loads(line); done[(r["sid"], r["order_id"])] = r
    scen = load_patil()[:args.limit]
    gts = {s.sid: set(s.gt) for s in scen}
    if args.report_only:
        report(list(done.values()), gts); return
    backend = make_backend(args.backend, args.model)
    obs = Observer(backend, r=args.r)
    for s in scen:
        turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
        z = s.secret; n = len(turns)
        rng = random.Random(args.seed * 100003 + hash(s.sid) % 100000)
        orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(args.orders)]
        for oid, order in enumerate(orders):
            if (s.sid, oid) in done:
                continue
            V = lambda S: obs.judge_excerpt(render(turns, S), z)
            qc = QueryCounter(); S = backward_search_order(V, order, qc)
            rec = {"sid": s.sid, "order_id": oid, "order": order, "q": qc.n}
            if S is not None and len(S) >= 2:
                ok, cd = certify(obs, turns, S, z, detail=True)
                rec.update({"S": sorted(S), "jac": jaccard(S, s.gt), "cert": bool(ok),
                            "cd": {k: bool(cd[k]) for k in ("base", "single", "MR1", "MR3", "MR4")}})
            elif S is not None:
                rec.update({"S": sorted(S), "jac": jaccard(S, s.gt), "cert": False, "single_cue": True})
            os.makedirs(os.path.dirname(prog), exist_ok=True)
            with open(prog, "a", encoding="utf8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(s.sid, oid)] = rec
            if args.sleep: time.sleep(args.sleep)
        print(f"  {s.sid}: {len(orders)} orders done")
    report(list(done.values()), gts)


if __name__ == "__main__":
    main()
