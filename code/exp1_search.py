# -*- coding: utf-8 -*-
"""
================================================================================
 Experiment 1 — correctness, robustness and scalability of the search algorithm  (standalone)
================================================================================
 Observer: a 'controlled oracle'. This is not a mock LLM but a deliberate device for checking against the ground truth.
   For the ground-truth minimal set gt, V(S)=1 ⇔ gt⊆S. To mimic the stochastic errors of a real LLM,
   each verdict is flipped with probability ε (noise injection), and an r-vote majority is used as mitigation.
 Data: Patil (actual files) + HotpotQA/MuSiQue (when actual file paths are given). Only the combinatorial structure is used, not the text.
 Usage:
   python3 exp1_search.py
   python3 exp1_search.py --hotpotqa data/hotpot_dev_distractor_v1.json \
                          --musique  data/musique_ans_v1.0_dev.jsonl
================================================================================
"""
import argparse, random, statistics as st
from datasets import load_patil, load_hotpotqa, load_musique
from search import backward_search, jaccard, QueryCounter


class OracleObserver:
    """Controlled oracle. gt-based verdict + noise (ε) + majority vote (r). (Not an LLM)"""
    def __init__(self, gt, eps=0.0, r=1, rng=None):
        self.gt = set(gt); self.eps = eps; self.r = r
        self.rng = rng or random.Random(0)

    def __call__(self, S):
        S = set(S); votes = 0
        for _ in range(self.r):
            t = 1 if self.gt.issubset(S) else 0
            if self.rng.random() < self.eps:
                t = 1 - t
            votes += t
        return 1 if votes * 2 >= self.r else 0


def run_dataset(name, scen, seeds=(0, 1, 2, 3, 4)):
    # Noise levels: (label, ε, r)
    LEVELS = [("무잡음", 0.00, 1), ("약", 0.05, 5), ("중", 0.10, 7), ("강", 0.20, 9)]
    us = [len(s.units) for s in scen]
    print("=" * 70)
    print(f"[{name}] 시나리오 {len(scen)}개 · 단서 평균 {st.mean(us):.1f}개 "
          f"· 정답 평균 {st.mean([len(s.gt) for s in scen]):.1f}개")
    print("-" * 70)
    print("  잡음(ε,r)   | 정확 복원율(평균±표준편차) | 자카드(평균) | 질의수")
    for label, eps, r in LEVELS:
        exact, jac, q = [], [], []
        for sd in seeds:
            rng = random.Random(sd); e = 0; jj = []; qq = []
            for s in scen:
                obs = OracleObserver(s.gt, eps=eps, r=r,
                                     rng=random.Random(rng.randint(0, 10**9)))
                qc = QueryCounter()
                found = backward_search(obs, list(range(len(s.units))), qc)
                qq.append(qc.n)
                found = found or frozenset()
                jj.append(jaccard(found, s.gt))
                if set(found) == set(s.gt): e += 1
            exact.append(e); jac.append(st.mean(jj)); q.append(st.mean(qq))
        sd_e = st.stdev(exact) if len(exact) > 1 else 0.0
        print(f"  {label:<4}(ε={eps:.2f},r={r}) | {st.mean(exact):6.1f} ± {sd_e:4.1f} / {len(scen):<4}"
              f"| {st.mean(jac):.3f}       | {st.mean(q):.0f}")
    # Baseline: random selection given the ground-truth size (lower bound on performance)
    def rnd(seed):
        rng = random.Random(seed)
        return st.mean([jaccard(set(rng.sample(range(len(s.units)), len(s.gt))), s.gt) for s in scen])
    print(f"  [베이스라인] 무작위(크기 제공) 자카드 = {st.mean([rnd(sd) for sd in seeds]):.3f}"
          f" · 전수 질의수 = {st.mean([2**len(s.units) for s in scen]):.0f} "
          f"(제안: {st.mean([len(s.units)+1 for s in scen]):.0f})")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patil", default=None, help="Patil data 디렉터리(기본: 환경변수/기본경로)")
    ap.add_argument("--hotpotqa", default=None, help="hotpot_dev_distractor_v1.json 경로")
    ap.add_argument("--musique", default=None, help="musique_ans_*.jsonl 경로")
    args = ap.parse_args()

    run_dataset("Patil (프라이버시)", load_patil(args.patil))
    if args.hotpotqa:
        run_dataset("HotpotQA (2홉)", load_hotpotqa(args.hotpotqa))
    if args.musique:
        run_dataset("MuSiQue (2–4홉)", load_musique(args.musique, limit=500))
    if not (args.hotpotqa or args.musique):
        print("※ HotpotQA·MuSiQue 는 --hotpotqa / --musique 로 실 파일 경로를 지정하면 함께 실행됩니다.")
