# -*- coding: utf-8 -*-
"""
exp3_compare.py — compares the Experiment 3-1 results of the observer models in one table.
Reads every e2e_progress_*.jsonl (the progress files written by exp3_endtoend.py) in the same folder.

  python3 exp3_compare.py                 # all scenarios
  python3 exp3_compare.py --sids common   # only scenarios completed by every model (fair comparison)
"""
import glob, json, argparse, statistics as st, re, os
from collections import Counter
from math import sqrt

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z*sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line); out[r["sid"]] = r
    return out

def summarize(done):
    abc = Counter(r["abc"] for r in done.values())
    n = len(done)
    paths = [r for r in done.values() if r["abc"] != "C" and r.get("jac") is not None]
    jac = st.mean(r["jac"] for r in paths) if paths else float("nan")
    mono = [r["mono"] for r in done.values() if r.get("mono") is not None]
    ct = len(paths); cp = sum(int(r["cert"]) for r in paths); c4 = sum(int(r["cert4"]) for r in paths)
    rel = {k: (sum(int(r["cd"][k]) for r in paths) / ct if ct else float("nan"))
           for k in ("base", "single", "MR1", "MR3", "MR4")}
    b_ask = [r["b_ask"] for r in done.values() if r.get("b_ask") is not None]
    b_loo = [r["b_loo"] for r in done.values() if r.get("b_loo") is not None]
    return dict(n=n, A=abc["A"], B=abc["B"], C=abc["C"],
                elic=abc["A"]/n if n else float("nan"), elic_ci=wilson(abc["A"], n),
                paths=ct, jac=jac, mono=st.mean(mono) if mono else float("nan"),
                cert=cp, cert_rate=cp/ct if ct else float("nan"), cert_ci=wilson(cp, ct),
                cert4_rate=c4/ct if ct else float("nan"), rel=rel,
                b_ask=st.mean(b_ask) if b_ask else float("nan"),
                b_loo=st.mean(b_loo) if b_loo else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", choices=["all", "common"], default="all")
    ap.add_argument("--glob", default="e2e_progress_*.jsonl", help='e.g. "e2e_progress_*_mr4fix.jsonl"')
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    if not files:
        print("No e2e_progress_*.jsonl found. Run exp3_endtoend.py first."); return
    runs = {}
    for f in files:
        m = re.match(r"e2e_progress_(.+?)_(.+)_r(\d+)(_\w+)?\.jsonl", os.path.basename(f))
        label = f"{m.group(1)}:{m.group(2)} (r={m.group(3)})" if m else f
        runs[label] = load(f)
    if a.sids == "common":
        common = set.intersection(*(set(d) for d in runs.values()))
        runs = {k: {s: v[s] for s in common} for k, v in runs.items()}
        print(f"Note: comparing only the {len(common)} scenarios completed by every model\n")

    S = {k: summarize(v) for k, v in runs.items()}
    cols = list(S)
    def row(name, fn):
        print(f"{name:<22}" + "".join(f"{fn(S[c]):>26}" for c in cols))
    print(f"{'':<22}" + "".join(f"{c[:25]:>26}" for c in cols))
    print("-" * (22 + 26 * len(cols)))
    row("scenarios n", lambda s: f"{s['n']}")
    row("induction A/B/C", lambda s: f"{s['A']}/{s['B']}/{s['C']}")
    row("induction recall (A rate)", lambda s: f"{s['elic']:.3f} [{s['elic_ci'][0]:.2f},{s['elic_ci'][1]:.2f}]")
    row("paths found", lambda s: f"{s['paths']}")
    row("path Jaccard", lambda s: f"{s['jac']:.3f}")
    row("monotonicity violation", lambda s: f"{s['mono']:.4f}")
    row("validation passed (AND)", lambda s: f"{s['cert']}/{s['paths']} = {s['cert_rate']:.3f}")
    row("  95% CI", lambda s: f"[{s['cert_ci'][0]:.2f}, {s['cert_ci'][1]:.2f}]")
    row("relaxed (4 of 5)", lambda s: f"{s['cert4_rate']:.3f}")
    for k in ("base", "single", "MR1", "MR3", "MR4"):
        row(f"  pass rate {k}", lambda s, k=k: f"{s['rel'][k]:.2f}")
    row("ask-the-model Jaccard", lambda s: f"{s['b_ask']:.3f}")
    row("leave-one-out", lambda s: f"{s['b_loo']:.3f}")
    print("\nReading: if induction and validation pass rates differ by model, the claim that judgment instability depends on the observer is supported.")
    print("         If MR3/MR4 pass rates rise for stronger models, the bottleneck lies with the observer, not the method.")

if __name__ == "__main__":
    main()
