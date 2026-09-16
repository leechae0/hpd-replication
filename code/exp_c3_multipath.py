# -*- coding: utf-8 -*-
"""
exp_c3_multipath.py — Appendix C.3: finding multiple paths and minimum blocking (controlled setting).
Produces Tables C.2 and C.3 of the paper. No LLM is used.

Judge: the rule-based oracle of Experiment 1 — V(S) = 1 iff some planted path is a subset of S —
with optional judgment error ε (each vote flips independently) and r-round majority voting.

Datasets
  synthetic : |U| ∈ {8, 12}; k ∈ {1,2,3,4} planted inclusion-minimal paths of size 2–4 drawn from the same
              cue pool (so they may overlap); 50 cases per (|U|, k) → 400 cases. Rows of the tables aggregate over |U|.
  patil     : the 116 Patil scenarios; 1 or 2 ground-truth cues are restated as additional utterances,
              which yields 2 or 4 planted paths (every combination of original/duplicate); 232 cases.

Per case and seed
  single   : one backward search (Algorithm 1)                      → S1 or none
  re-search: a second backward search with the cues of S1 hidden    → S2 or none  (as in Appendix C.2)
  full     : complete enumeration of minimal sufficient sets (search.enumerate_all)
  blocking : optimal hitting set (exhaustive) and greedy hitting set over the enumerated paths;
             re-inference = error-free judgment on U ∖ B for three blocking rules:
             first-path-only (hide one cue of S1), random (|B_opt| random cues), all-paths (greedy B).

  python3 exp_c3_multipath.py            # prints both tables; also writes results/c3_results.json
"""
import argparse, itertools, json, os, random, statistics as st
from collections import Counter
from datasets import load_patil
from search import backward_search, enumerate_all, QueryCounter

SEEDS = (0, 1, 2, 3, 4)
CONDS = [("none", 0.0, 1), ("20%", 0.20, 9)]


class Oracle:
    def __init__(self, paths, eps=0.0, r=1, rng=None):
        self.paths = [frozenset(p) for p in paths]; self.eps = eps; self.r = r
        self.rng = rng or random.Random(0)

    def truth(self, S):
        S = set(S)
        return 1 if any(p <= S for p in self.paths) else 0

    def __call__(self, S):
        t = self.truth(S); votes = 0
        for _ in range(self.r):
            v = t
            if self.eps and self.rng.random() < self.eps:
                v = 1 - v
            votes += v
        return 1 if votes * 2 >= self.r else 0


# ------------------------------------------------------------ cases
def synthetic_cases(rng, n_per=50):
    cases = []
    for n in (8, 12):
        for k in (1, 2, 3, 4):
            made = 0
            while made < n_per:
                paths = []
                ok = True
                for j in range(k):
                    for _ in range(200):
                        size = rng.randint(2, 4)
                        p = frozenset(rng.sample(range(n), size))   # paths drawn from the same cue pool may overlap
                        if all(not (p <= q or q <= p) for q in paths):   # inclusion-minimal family
                            paths.append(p); break
                    else:
                        ok = False; break
                if ok:
                    cases.append({"group": f"synthetic·{k}", "n": n, "paths": paths}); made += 1
    return cases


def patil_cases(rng):
    cases = []
    for s in load_patil():
        n0 = len(s.units); G = sorted(s.gt)
        for dup in (1, 2):
            chosen = rng.sample(G, dup)
            n = n0 + dup
            alt = {c: n0 + i for i, c in enumerate(chosen)}      # duplicate cue index for each restated cue
            paths = []
            for choice in itertools.product([0, 1], repeat=dup):
                p = set(G)
                for c, use_dup in zip(chosen, choice):
                    if use_dup:
                        p.remove(c); p.add(alt[c])
                paths.append(frozenset(p))
            cases.append({"group": f"patil·{2 ** dup}", "n": n, "paths": paths})
    return cases


# ------------------------------------------------------------ blocking
def optimal_hitting_set(paths, n):
    if not paths: return set()
    for size in range(1, n + 1):
        for B in itertools.combinations(range(n), size):
            Bs = set(B)
            if all(p & Bs for p in paths):
                return Bs
    return set(range(n))


def greedy_hitting_set(paths):
    B, remaining = set(), [set(p) for p in paths]
    while remaining:
        cnt = Counter(u for p in remaining for u in p)
        best = max(sorted(cnt), key=lambda u: cnt[u])
        B.add(best); remaining = [p for p in remaining if best not in p]
    return B


# ------------------------------------------------------------ one case
def run_case(case, eps, r, seed):
    n = case["n"]; planted = set(case["paths"]); U = list(range(n))
    rng = random.Random(seed * 1_000_003 + hash(tuple(sorted(map(tuple, planted)))) % 100_000)
    V = Oracle(planted, eps, r, rng=random.Random(rng.randint(0, 10 ** 9)))
    truth = Oracle(planted)                       # error-free judge for re-inference
    out = {}
    # single search
    qc1 = QueryCounter(); S1 = backward_search(V, U, qc1)
    out["q_single"] = qc1.n
    if S1 is not None and len(S1) < 2: S1 = None          # single-cue check: sets of size < 2 are not paths
    found_single = [S1] if S1 else []
    # re-search with S1's cues hidden (Appendix C.2 procedure)
    found_re = list(found_single)
    if S1:
        rest = [u for u in U if u not in S1]
        S2 = backward_search(V, rest, QueryCounter()) if len(rest) >= 2 else None
        if S2 and len(S2) >= 2: found_re.append(S2)
    # full enumeration
    qcf = QueryCounter(); M = enumerate_all(V, U, qcf)
    M = [frozenset(m) for m in M if len(m) >= 2]           # single-cue check applied to enumerated sets
    out["q_full"] = qcf.n
    out["n_single"] = len(found_single); out["n_re"] = len(found_re); out["n_full"] = len(M)
    out["full_recovery"] = int(set(M) == planted)
    out["no_path_left"] = int(planted <= set(M))          # nothing undiscovered
    # blocking
    B_opt = optimal_hitting_set(M, n); B_gr = greedy_hitting_set(M)
    out["b_opt"] = len(B_opt); out["b_greedy"] = len(B_gr)
    if S1:
        B_first = {min(S1)}
    else:
        B_first = set()
    B_rand = set(rng.sample(U, len(B_opt))) if B_opt else set()
    out["re_first"] = truth.truth(set(U) - B_first)
    out["re_random"] = truth.truth(set(U) - B_rand)
    out["re_all"] = truth.truth(set(U) - B_gr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "c3_results.json"))
    a = ap.parse_args()
    cases = synthetic_cases(random.Random(2026)) + patil_cases(random.Random(2026))
    groups = ["synthetic·1", "synthetic·2", "synthetic·3", "synthetic·4", "patil·2", "patil·4"]
    print(f"cases: {len(cases)}  ({Counter(c['group'] for c in cases)})")
    keys = ["q_single", "q_full", "n_single", "n_re", "n_full", "full_recovery", "no_path_left",
            "b_opt", "b_greedy", "re_first", "re_random", "re_all"]
    table = {}
    for label, eps, r in CONDS:
        for g in groups:
            acc = {k: [] for k in keys}; agree = []
            for seed in SEEDS:
                for c in cases:
                    if c["group"] != g: continue
                    o = run_case(c, eps, r, seed)
                    for k in keys: acc[k].append(o[k])
                    agree.append(int(o["b_opt"] == o["b_greedy"]))
            row = {k: st.mean(v) for k, v in acc.items()}; row["greedy_agree"] = st.mean(agree)
            table[(label, g)] = row
    # ---- print Table C.2
    print("\nTABLE C.2  Multiple-path discovery")
    print(f"{'scenario·paths':<14}{'error':<8}{'paths found single/re/full':<30}{'full rec.':>10}{'no left':>9}{'queries s/f':>14}")
    for label, eps, r in CONDS:
        for g in groups:
            w = table[(label, g)]
            print(f"{g:<14}{label:<8}{w['n_single']:.2f} / {w['n_re']:.2f} / {w['n_full']:.2f}{'':<12}"
                  f"{w['full_recovery']:>10.3f}{w['no_path_left']:>9.3f}{w['q_single']:>7.0f} / {w['q_full']:<5.0f}")
    print("\nTABLE C.3  Re-inference by blocking method")
    print(f"{'scenario·paths':<14}{'error':<8}{'cues hidden opt/greedy':<26}{'first':>8}{'random':>8}{'all':>8}{'agree':>8}")
    for label, eps, r in CONDS:
        for g in groups:
            w = table[(label, g)]
            print(f"{g:<14}{label:<8}{w['b_opt']:.2f} / {w['b_greedy']:.2f}{'':<12}"
                  f"{w['re_first']:>8.3f}{w['re_random']:>8.3f}{w['re_all']:>8.3f}{w['greedy_agree']:>8.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({f"{l}|{g}": v for (l, g), v in table.items()}, open(a.out, "w"), indent=1)
    print("\nwritten:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
