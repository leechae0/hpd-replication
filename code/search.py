# -*- coding: utf-8 -*-
"""
search.py — 탐색 알고리즘 (실코드). 관찰자 판정 함수 V(S)->{0,1} 위에서 동작.
 - backward_search : 전체에서 각 단서를 한 번씩 제거해 최소 지지집합 1개를 복원. |U|+1 질의.
 - enumerate_all   : 모든 최소 지지집합을 완전 열거(단조 쌍대생성). 소규모용.
"""
import itertools


class QueryCounter:
    def __init__(self): self.n = 0
    def tick(self): self.n += 1


def backward_search(V, universe, qc):
    """전체 집합에서 시작해 각 단서 제거 시 결론이 유지되면 버린다(=MR2). 최소 지지집합 반환."""
    qc.tick()
    if V(list(universe)) == 0:
        return None
    S = set(universe)
    for i in list(universe):
        qc.tick()
        if V(list(S - {i})) == 1:
            S = S - {i}
    return frozenset(S)


# ------------- 완전 열거 (모든 최소 지지집합) : 단조 쌍대생성 -------------
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
    """모든 최소 지지집합을 완전 열거. (단서 수가 작을 때만 사용)"""
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
