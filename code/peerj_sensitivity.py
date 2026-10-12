# -*- coding: utf-8 -*-
"""
peerj_sensitivity.py: Table S6 of the PeerJ CS supplement. Recomputes the main results after
excluding the 12 scenarios whose benchmark secret the human induction audit marked as
contradicted by or unsupported by the scenario records (note = benchmark-label-inconsistent
in results/induction/annotation_sheet.csv). No API calls.

Usage (from code/):  python3 peerj_sensitivity.py [--json ../results/peerj_sensitivity.json]
"""
import argparse, csv, json, math, os, random, statistics as st
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "results", "logs")
SHEET = os.path.join(HERE, "..", "results", "induction", "annotation_sheet.csv")
OBS = [("GPT-4o-mini", "openai_gpt-4o-mini-2024-07-18"),
       ("Qwen3.6-27B", "openrouter_qwen_qwen3.6-27b"),
       ("Gemini 2.5 Flash-Lite", "openrouter_google_gemini-2.5-flash-lite")]


def load(tag):
    rows = {}
    with open(os.path.join(LOGS, f"e2e_progress_{tag}_r5_mr4fix.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); rows[r["sid"]] = r
    return rows


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 2), round(min(1, c + h), 2)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(rows, keep):
    rs = [rows[s] for s in keep]
    paths = [r for r in rs if r.get("S") and r.get("abc") in ("A", "B")]
    k = len(paths)
    struct = sum(1 for r in paths if r["cd"]["base"] and r["cd"]["single"])
    allv = sum(1 for r in paths if r.get("cert"))
    return {"scenarios": len(rs), "paths_returned": k, "path_rate": round(k / len(rs), 3),
            "mean_path_length": round(st.mean(len(r["S"]) for r in paths), 2),
            "agreement_jaccard": round(st.mean(r["jac"] for r in paths), 3),
            "both_structural": round(struct / k, 3),
            "MR1": round(sum(r["cd"]["MR1"] for r in paths) / k, 2),
            "MR3": round(sum(r["cd"]["MR3"] for r in paths) / k, 2),
            "MR4": round(sum(r["cd"]["MR4"] for r in paths) / k, 2),
            "all_five": round(allv / k, 3), "all_five_ci": wilson(allv, k), "validated_scenarios": allv,
            "removed_by_invariance": round((struct - allv) / struct, 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    sheet = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    flawed = sorted({r["sid"] for r in sheet if r.get("note", "").strip() == "benchmark-label-inconsistent"},
                    key=lambda s: int(s.split("_")[1]))
    data = {name: load(tag) for name, tag in OBS}
    allsids = sorted(set.intersection(*(set(d) for d in data.values())))
    keep = [s for s in allsids if s not in flawed]
    out = {"excluded": flawed, "n_all": len(allsids), "n_kept": len(keep), "observers": {}}
    for name, _ in OBS:
        out["observers"][name] = {"all": summarize(data[name], allsids), "kept": summarize(data[name], keep)}
    pairs = {}
    for (n1, _), (n2, _) in combinations(OBS, 2):
        v1 = {s: bool(data[n1][s].get("cert")) for s in keep}; v2 = {s: bool(data[n2][s].get("cert")) for s in keep}
        b = sum(1 for s in keep if v1[s] and not v2[s]); c = sum(1 for s in keep if v2[s] and not v1[s])
        pairs[f"{n1} vs {n2}"] = {"b": b, "c": c, "p_exact": round(mcnemar_exact(b, c), 4)}
    out["mcnemar_kept_unadjusted"] = pairs
    print("excluded:", ", ".join(flawed))
    for name, d in out["observers"].items():
        print(f"\n{name}")
        for kk in d["all"]:
            print(f"  {kk:<22} {str(d['all'][kk]):>14}  ->  {d['kept'][kk]}")
    print("\nMcNemar on kept scenarios (exact, unadjusted):")
    for kk, v in pairs.items():
        print(f"  {kk}: b={v['b']} c={v['c']} p={v['p_exact']}")
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), indent=2, ensure_ascii=False); print("wrote", a.json)


if __name__ == "__main__":
    main()
