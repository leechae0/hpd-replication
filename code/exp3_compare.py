# -*- coding: utf-8 -*-
"""
exp3_compare.py — 관찰자 모델별 실험 3-1 결과를 한 표로 비교한다.
같은 폴더의 e2e_progress_*.jsonl (exp3_endtoend.py가 남기는 진행 파일)을 모두 읽는다.

  python3 exp3_compare.py                 # 전부
  python3 exp3_compare.py --sids common   # 모든 모델이 공통으로 완료한 시나리오만(공정 비교)
"""
import glob, json, argparse, statistics as st, re
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
    a = ap.parse_args()
    files = sorted(glob.glob("e2e_progress_*.jsonl"))
    if not files:
        print("e2e_progress_*.jsonl 이 없습니다. exp3_endtoend.py 를 먼저 돌리세요."); return
    runs = {}
    for f in files:
        m = re.match(r"e2e_progress_(.+?)_(.+)_r(\d+)\.jsonl", f)
        label = f"{m.group(1)}:{m.group(2)} (r={m.group(3)})" if m else f
        runs[label] = load(f)
    if a.sids == "common":
        common = set.intersection(*(set(d) for d in runs.values()))
        runs = {k: {s: v[s] for s in common} for k, v in runs.items()}
        print(f"※ 모든 모델이 공통으로 완료한 {len(common)}개 시나리오만 비교\n")

    S = {k: summarize(v) for k, v in runs.items()}
    cols = list(S)
    def row(name, fn):
        print(f"{name:<22}" + "".join(f"{fn(S[c]):>26}" for c in cols))
    print(f"{'':<22}" + "".join(f"{c[:25]:>26}" for c in cols))
    print("-" * (22 + 26 * len(cols)))
    row("시나리오 n", lambda s: f"{s['n']}")
    row("유도 A/B/C", lambda s: f"{s['A']}/{s['B']}/{s['C']}")
    row("유도 재현율(A율)", lambda s: f"{s['elic']:.3f} [{s['elic_ci'][0]:.2f},{s['elic_ci'][1]:.2f}]")
    row("경로 확보", lambda s: f"{s['paths']}")
    row("경로 자카드", lambda s: f"{s['jac']:.3f}")
    row("단조성 위반율", lambda s: f"{s['mono']:.4f}")
    row("검증 통과(AND)", lambda s: f"{s['cert']}/{s['paths']} = {s['cert_rate']:.3f}")
    row("  95% CI", lambda s: f"[{s['cert_ci'][0]:.2f}, {s['cert_ci'][1]:.2f}]")
    row("완화(5중 4)", lambda s: f"{s['cert4_rate']:.3f}")
    for k in ("base", "single", "MR1", "MR3", "MR4"):
        row(f"  불변율 {k}", lambda s, k=k: f"{s['rel'][k]:.2f}")
    row("직접질문 자카드", lambda s: f"{s['b_ask']:.3f}")
    row("leave-one-out", lambda s: f"{s['b_loo']:.3f}")
    print("\n읽는 법: 유도·검증 통과율이 모델마다 다르면 '판정 불안정은 관찰자 의존'(VI절) 주장이 실측으로 뒷받침됨.")
    print("        MR3·MR4 불변율이 상위 모델에서 오르면 병목이 방법이 아니라 관찰자 쪽임을 보임.")

if __name__ == "__main__":
    main()
