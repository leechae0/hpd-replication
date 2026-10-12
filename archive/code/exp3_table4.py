# -*- coding: utf-8 -*-
"""
exp3_table4.py: message-level versus path-level auditing with the benchmark secret fixed (Table 4).

For every dialogue the target secret z is fixed (a negative is judged against the secret of the
scenario it was built from), and two audits are compared on the same observer:

  message-level : flag the dialogue if any single utterance u gives V_z({u}) = 1
  path-level    : flag the dialogue if HPD returns a validated path (discovery + all five checks)

Dialogues
  positives      : the 116 Patil scenarios. The path-level result is read from the main-run log
                   results/logs/e2e_progress_<tag>_r5_mr4fix.jsonl (no new calls).
  hard negatives : scenario i with its annotated cue combination removed (first --neg-n scenarios),
                   judged against scenario i's secret.
  mixed negatives: 3 to 5 cues drawn from different scenarios, judged against the secret of one
                   source scenario whose annotated combination is not fully present.

Every judgment uses the same Observer, prompt, value-exposed rendering, and r = 5 majority vote as
the main run. Judgments are cached per (dialogue, cue subset, secret), so the run is resumable.

Run (keys only via export in the terminal; never commit keys):
  export OPENAI_API_KEY=...        ; export OPENROUTER_API_KEY=...
  python3 code/exp3_table4.py --backend openai     --model gpt-4o-mini-2024-07-18
  python3 code/exp3_table4.py --backend openrouter --model qwen/qwen3.6-27b
  python3 code/exp3_table4.py --backend openrouter --model google/gemini-2.5-flash-lite
  python3 code/exp3_table4.py --summary            # all three observers -> results/table4_summary.json

Cost per observer (r = 5): about 116 x 8 x 5 = 4,640 calls for the message-level audit of the
positives, plus about 1,500 to 3,000 calls for the 45 negatives. Run the three observers in three
terminals at once if the rate limits allow it.
"""
import argparse, glob, hashlib, json, math, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue, Scenario
from search import QueryCounter
from exp3_endtoend import render, find_path, certify

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOGS = os.path.join(ROOT, "results", "logs")
SUMMARY = os.path.join(ROOT, "results", "table4_summary.json")
SEED = 20261006


def tag_of(backend, model):
    return f"{backend}_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}"


# ------------------------------------------------------------------ dialogues
def build_dialogues(neg_n, n_mixed):
    pos = load_patil()
    dial = [("pos", s.sid, s, s.secret) for s in pos]
    for s in pos[:neg_n]:
        keep = [dict(u) for i, u in enumerate(s.units) if i not in s.gt]
        if len(keep) >= 1:
            dial.append(("hard", f"hard_{s.sid}", Scenario(sid=f"hard_{s.sid}", units=keep, secret="", gt=frozenset()), s.secret))
    rng = random.Random(SEED)
    src = pos[:neg_n]
    made = 0
    while made < n_mixed:
        k = rng.randint(3, 5)
        picks = rng.sample([(si, ui) for si, s in enumerate(src) for ui in range(len(s.units))], k)
        if len({si for si, _ in picks}) < 2:
            continue
        owner = rng.choice(sorted({si for si, _ in picks}))
        owner_cues = {ui for si, ui in picks if si == owner}
        if src[owner].gt <= owner_cues:          # would contain the full sensitive combination
            continue
        units = [dict(src[si].units[ui]) for si, ui in picks]
        sid = f"mixed_{made:02d}"
        dial.append(("mixed", sid, Scenario(sid=sid, units=units, secret="", gt=frozenset()), src[owner].secret))
        made += 1
    return dial


# ------------------------------------------------------------------ cached observer
class Cached:
    def __init__(self, obs, path):
        self.obs, self.path, self.cache, self.sent = obs, path, {}, 0
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line); self.cache[d["k"]] = d["v"]

    def judge(self, excerpt, z):
        k = hashlib.sha1((excerpt + "\x00" + z).encode("utf-8")).hexdigest()[:16]
        if k not in self.cache:
            v = self.obs.judge_excerpt(excerpt, z); self.sent += 1
            self.cache[k] = v
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        return self.cache[k]

    def judge_excerpt(self, excerpt, z):      # same interface as Observer, so find_path/certify reuse it
        return self.judge(excerpt, z)


# ------------------------------------------------------------------ run one observer
def run(args):
    model = args.model
    tag = tag_of(args.backend, model)
    main_log = os.path.join(LOGS, f"e2e_progress_{tag}_r5_mr4fix.jsonl")
    if not os.path.exists(main_log):
        sys.exit(f"main-run log not found: {main_log}")
    main = {}
    for line in open(main_log, encoding="utf-8"):
        if line.strip():
            d = json.loads(line); main[d["sid"]] = d
    prog = os.path.join(LOGS, f"table4_{tag}.jsonl")
    done = {}
    if os.path.exists(prog):
        for line in open(prog, encoding="utf-8"):
            if line.strip():
                d = json.loads(line); done[d["sid"]] = d

    backend = make_backend(args.backend, model)
    obs = Cached(Observer(backend, r=args.r), os.path.join(LOGS, f"table4_cache_{tag}.jsonl"))
    dial = build_dialogues(args.neg_n, args.mixed_n)
    todo = [d for d in dial if d[1] not in done]
    print(f"[{tag}] dialogues {len(dial)} (pos {sum(d[0]=='pos' for d in dial)}, hard {sum(d[0]=='hard' for d in dial)}, "
          f"mixed {sum(d[0]=='mixed' for d in dial)}); done {len(done)}; to run {len(todo)}")
    t0 = time.time()
    for n, (kind, sid, scen, z) in enumerate(todo, 1):
        try:
            turns = scenario_to_dialogue(None, scen, use_llm=False)
            flags = [obs.judge(render(turns, [i]), z) for i in range(len(turns))]
            rec = {"sid": sid, "kind": kind, "msg_flags": flags, "msg": any(flags)}
            if kind == "pos":
                m = main.get(sid, {})
                rec["hpd"] = bool(m.get("cert", False)); rec["hpd_source"] = "main_run"
            else:
                S = find_path(obs, turns, z, QueryCounter())
                if S is not None and len(S) >= 2:
                    ok, cd = certify(obs, turns, S, z, detail=True)
                    rec["hpd"] = bool(ok); rec["S"] = sorted(S)
                    rec["cd"] = {k: bool(cd[k]) for k in ("base", "single", "MR1", "MR3", "MR4")}
                else:
                    rec["hpd"] = False; rec["S"] = sorted(S) if S is not None else None
            with open(prog, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[sid] = rec
        except Exception as e:
            print(f"  [stopped at {sid}] {str(e)[:160]}\n  -> rerun the same command to resume."); break
        if n % 10 == 0:
            print(f"  ...{n}/{len(todo)}  calls sent {obs.sent * args.r}  ({(time.time()-t0)/60:.1f} min)")
    summarize_one(tag, list(done.values()))


# ------------------------------------------------------------------ summary
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def summarize_one(tag, rows):
    out = {}
    for kind in ("pos", "hard", "mixed"):
        rs = [r for r in rows if r["kind"] == kind]
        n = len(rs)
        for audit in ("msg", "hpd"):
            k = sum(1 for r in rs if r.get(audit))
            out[f"{kind}_{audit}"] = {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci": wilson(k, n)}
    pos = [r for r in rows if r["kind"] == "pos"]
    out["pos_hpd_only"] = sum(1 for r in pos if r.get("hpd") and not r.get("msg"))
    out["pos_msg_only"] = sum(1 for r in pos if r.get("msg") and not r.get("hpd"))
    out["pos_both"] = sum(1 for r in pos if r.get("msg") and r.get("hpd"))
    out["pos_neither"] = sum(1 for r in pos if not r.get("msg") and not r.get("hpd"))
    print(f"\n[{tag}]")
    for kind, label in (("pos", "positives"), ("hard", "combination removed"), ("mixed", "mixed")):
        a, b = out[f"{kind}_msg"], out[f"{kind}_hpd"]
        print(f"  {label:20s} message-level {a['k']}/{a['n']} = {a['rate']} {a['ci']}   path-level {b['k']}/{b['n']} = {b['rate']} {b['ci']}")
    print(f"  positives: path-level only {out['pos_hpd_only']}, message-level only {out['pos_msg_only']}, "
          f"both {out['pos_both']}, neither {out['pos_neither']}")
    return out


def summary_all():
    res = {}
    for p in sorted(glob.glob(os.path.join(LOGS, "table4_*.jsonl"))):
        if "_cache_" in p:
            continue
        tag = os.path.basename(p)[len("table4_"):-len(".jsonl")]
        rows = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                d = json.loads(line); rows[d["sid"]] = d
        res[tag] = summarize_one(tag, list(rows.values()))
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {SUMMARY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "openrouter", "groq", "anthropic", "gemini", "cerebras"])
    ap.add_argument("--model")
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--neg-n", type=int, default=30, help="hard negatives from the first N scenarios")
    ap.add_argument("--mixed-n", type=int, default=15)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary_all()
    elif a.backend and a.model:
        run(a)
    else:
        ap.error("give --backend and --model, or --summary")
