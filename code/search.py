# -*- coding: utf-8 -*-
"""
search.py — search algorithms (actual code). They operate on the observer verdict function V(S)->{0,1}.
 - backward_search : starting from the full set, remove each cue once to recover one minimal support set. |U|+1 queries.
 - enumerate_all   : exhaustively enumerate all minimal support sets (monotone dualization). For small instances.
"""
import itertools


class QueryCounter:
    def __init__(self): self.n = 0
    def tick(self): self.n += 1


def backward_search(V, universe, qc):
    """Start from the full set and drop each cue whose removal keeps the conclusion (=MR2). Returns a minimal support set."""
    qc.tick()
    if V(list(universe)) == 0:
        return None
    S = set(universe)
    for i in list(universe):
        qc.tick()
        if V(list(S - {i})) == 1:
            S = S - {i}
    return frozenset(S)


# ------------- Exhaustive enumeration (all minimal support sets): monotone dualization -------------
def _minimize(V, X, qc):
    qc.tick()
    if V(list(X)) == 0: return None
    S = set(X)
    for i in list(X):
        qc.tick()
        if V(list(S - {i})) == 1: S = S - {i}
    return frozenset(S)

def _maximize(V, X, D, qc):
    qc.tick()
    if V(list(X)) == 1: return None
    F = set(X)
    for i in [d for d in D if d not in X]:
        qc.tick()
        if V(list(F | {i})) == 0: F = F | {i}
    return frozenset(F)

def _conflict(D, M, N):
    for k in range(len(D) + 1):
        for combo in itertools.combinations(sorted(D), k):
            X = frozenset(combo)
            if any(m <= X for m in M): continue
            if any(X <= n for n in N): continue
            return X
    return None

def enumerate_all(V, universe, qc, budget=10**6):
    """Exhaustively enumerate all minimal support sets. (Use only when the number of cues is small)"""
    D = set(universe); M, N = [], []
    while qc.n < budget:
        X = _conflict(D, M, N)
        if X is None:
            return M
        if V(list(X)) == 1:
            qc.tick(); S = _minimize(V, X, qc)
            if S is not None and S not in M: M.append(S)
        else:
            qc.tick(); T = _maximize(V, X, D, qc)
            if T is not None and T not in N: N.append(T)
    return M


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0
