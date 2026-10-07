# -*- coding: utf-8 -*-
"""
peerj_tables.py: recompute the numbers in Tables 2 and 3 and Section 4.4 of the PeerJ CS manuscript
from the logs in this repository. No API calls.

Inputs
  results/logs/e2e_progress_<tag>_r5_mr4fix.jsonl   main run, MR4 corrected (one record per scenario)
  results/induction/semantic_scores.json            human-judged induction (written by exp3_induction.py score)

Usage (from code/):
  python3 peerj_tables.py            # prints the tables
  python3 peerj_tables.py --json ../results/peerj_tables.json
"""
import argparse, json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "results", "logs")
SEM = os.path.join(HERE, "..", "results", "induction", "semantic_scores.json")
OBS = [("GPT-4o-mini", "openai_gpt-4o-mini-2024-07-18"),
       ("Qwen3.6-27B", "openrouter_qwen_qwen3.6-27b"),
       ("Gemini 2.5 Flash-Lite", "openrouter_google_gemini-2.5-flash-lite")]
CHECKS = ("base", "single", "MR1", "MR3", "MR4")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - h), 2), round(min(1, c + h), 2)


def load(tag):
    rows = {}
    with open(os.path.join(LOGS, f"e2e_progress_{tag}_r5_mr4fix.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); rows[r["sid"]] = r
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    sem = json.load(open(SEM, encoding="utf-8")) if os.path.exists(SEM) else {}
    out = {}
    for name, tag in OBS:
        rows = load(tag)
        n = len(rows)
        # a path is counted when the scenario passed the induction gate (A or B) and discovery returned S
        paths = [r for r in rows if r.get("S") and r.get("abc") in ("A", "B")]
        k = len(paths)
        rate = lambda c: sum(1 for r in paths if r["cd"][c]) / k
        struct = sum(1 for r in paths if r["cd"]["base"] and r["cd"]["single"])
        allv = sum(1 for r in paths if r.get("cert"))
        four = sum(1 for r in paths if r.get("cert4"))
        o = {
            "scenarios": n,
            "lexical_induction": round(sum(1 for r in rows if r.get("abc") == "A") / n, 3),
            "human_induction": {kk: sem[tag][kk] for kk in ("semantic", "ci95", "annotated")} if tag in sem else None,
            "paths_returned": k,
            "mean_path_length": round(st.mean(len(r["S"]) for r in paths), 2),
            "agreement_jaccard": round(st.mean(r["jac"] for r in paths), 3),
            "sufficiency": round(rate("base"), 2), "single_cue": round(rate("single"), 2),
            "both_structural": round(struct / k, 3),
            "MR1": round(rate("MR1"), 2), "MR3": round(rate("MR3"), 2), "MR4": round(rate("MR4"), 2),
            "all_five": round(allv / k, 3), "all_five_ci": wilson(allv, k), "all_five_of_116": allv,
            "removed_by_invariance": f"{struct - allv}/{struct} = {(struct - allv) / struct:.2f}",
            "four_of_five": round(four / k, 3),
            "monotonicity_probed_scenarios": sum(1 for r in rows if r.get("mono") is not None),
            "monotonicity_violation_mean": round(st.mean(r["mono"] for r in rows if r.get("mono") is not None), 4),
        }
        out[name] = o
    for name, o in out.items():
        print(f"\n{name}")
        for kk, v in o.items():
            print(f"  {kk:<28} {v}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
