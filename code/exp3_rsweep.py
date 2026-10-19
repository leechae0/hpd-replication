# -*- coding: utf-8 -*-
"""
================================================================================
 Experiment 3-4 (optional) — validation calibration sweep: r ∈ {1,3,5,9} × (full AND vs 4-of-5)
================================================================================
 Purpose: show empirically that the real-setting true-acceptance rate of 0.448 is a product of
 conservative observer calibration, not a defect of the method. Path search is run once with r=5;
 only the validation stage is re-judged on the same path with different r (no repeated search cost).

 ★ Resumable: per-scenario results are saved to rsweep_progress.jsonl.
   If interrupted, rerun the same command: completed scenarios are skipped and all are aggregated.
   To start over, delete rsweep_progress.jsonl.

 Usage:
   OPENAI_API_KEY=.. python3 exp3_rsweep.py --backend openai --model gpt-4o-mini \
       --limit 30 --raw --sleep 1
================================================================================
"""
import argparse, json, os, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import QueryCounter
from exp3_endtoend import render, find_path, certify

R_LIST = [1, 3, 5, 9]
KEYS = ("base", "single", "MR1", "MR3", "MR4")
PROG = "rsweep_progress.jsonl"


def load_progress():
    done = {}
    if os.path.exists(PROG):
        for line in open(PROG, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["sid"]] = rec
    return done


def save_record(rec):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def aggregate(done):
    tot = 0
    andp = {r_: 0 for r_ in R_LIST}
    p4 = {r_: 0 for r_ in R_LIST}
    rel = {r_: {k: 0 for k in KEYS} for r_ in R_LIST}
    for rec in done.values():
        if rec.get("no_path"):
            continue
        tot += 1
        for r_ in R_LIST:
            rr = rec["r"][str(r_)]
            andp[r_] += int(rr["and"])
            p4[r_] += int(rr["p4"])
            for k in KEYS:
                rel[r_][k] += int(rr["rel"][k])
    return tot, andp, p4, rel


def report(done):
    tot, andp, p4, rel = aggregate(done)
    n_skip = sum(1 for rec in done.values() if rec.get("no_path"))
    print("\n" + "=" * 64 + f"\nExperiment 3-4 results ({tot} with a path · {n_skip} without a path)\n" + "=" * 64)
    print(f"{'r':>3} | {'all five (AND)':>14} | {'four of five':>14} | base·single·MR1·MR3·MR4")
    for r_ in R_LIST:
        if tot:
            rl = " ".join(f"{rel[r_][k]/tot:.2f}" for k in KEYS)
            print(f"{r_:>3} | {andp[r_]:>3}/{tot} = {andp[r_]/tot:.3f} | {p4[r_]:>3}/{tot} = {p4[r_]/tot:.3f} | {rl}")
    print("\nReading: if the all-five pass rate rises with r, the 0.448 rate is a judgment-instability (calibration) problem,")
    print("as claimed. The gap between four-of-five and all-five is the effect size of relaxing the threshold.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model)
    scen = load_patil()[args.start:args.start + args.limit]
    done = load_progress()
    remain = [s for s in scen if s.sid not in done]
    print(f"Experiment 3-4 validation calibration sweep — {len(scen)-len(remain)} of {len(scen)} done, "
          f"{len(remain)} to run now (r={R_LIST})")

    for k, s in enumerate(remain):
        try:
            obs5 = Observer(backend, r=5)
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            z = s.secret
            qc = QueryCounter()
            S = find_path(obs5, turns, z, qc)      # search: once, with r fixed at 5
            if S is None or len(S) < 2:
                rec = {"sid": s.sid, "no_path": True}
                save_record(rec); done[s.sid] = rec
                if args.sleep: time.sleep(args.sleep)
                continue
            rrec = {}
            for r_ in R_LIST:                       # sweep r for validation only
                obs = Observer(backend, r=r_)
                ok, cd = certify(obs, turns, S, z, detail=True)
                rrec[str(r_)] = {"and": int(ok),
                                 "p4": int(sum(int(cd[kk]) for kk in KEYS) >= 4),
                                 "rel": {kk: int(cd[kk]) for kk in KEYS}}
            rec = {"sid": s.sid, "S": sorted(S), "r": rrec}
            save_record(rec); done[s.sid] = rec
            if (k + 1) % 3 == 0:
                print(f"  ...{k+1}/{len(remain)} (paths found so far "
                      f"{sum(1 for x in done.values() if not x.get('no_path'))})")
            if args.sleep: time.sleep(args.sleep)
        except Exception as e:
            print(f"  [stopped] error at {s.sid}: {str(e)[:100]}")
            print(f"  → aggregating the completed scenarios. Re-run the same command to resume.")
            break

    report(done)


if __name__ == "__main__":
    main()
