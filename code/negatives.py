# -*- coding: utf-8 -*-
"""
================================================================================
 negatives.py — 대화 수준 탐지(실험 3-2)용 음성(비누출) 시나리오 생성
================================================================================
 양성(Patil): 민감 조합이 존재해 비밀이 형성되는 대화.
 음성 3종:
   (i)  distractor_only : 민감 조합(gt) 단서만 제거하고 교란 단서만 남김 → 숨은 비밀 없음
        (겉보기는 누출 대화와 유사한 hard negative; 사전지식 오탐을 시험)
   (ii) shuffled        : 서로 다른 시나리오의 단서를 뒤섞어 일관된 비밀이 성립하지 않게 함
   (iii) natural        : 레이블이 없는 자유형 정상 대화(MAGPIE) — 탐지 스크립트에서 로드

 각 음성 시나리오는 Scenario(secret="", gt=frozenset())로, 양성과 동일하게
 scenario_to_dialogue()로 자연어 대화로 변환되어 파이프라인에 투입된다.
================================================================================
"""
import random
from datasets import Scenario


def distractor_only(s):
    """민감 조합(gt) 단서만 제거하고 교란 단서만 남긴 음성 시나리오."""
    keep = [u for i, u in enumerate(s.units) if i not in s.gt]
    if len(keep) < 2:
        return None
    return Scenario(sid=f"neg_distract_{s.sid}", units=[dict(u) for u in keep],
                    secret="", gt=frozenset())


def shuffled(scen, rng, size=4):
    """서로 다른 시나리오의 단서를 뒤섞어 일관된 비밀이 없는 음성 시나리오."""
    pool = [u for s in scen for u in s.units]
    if len(pool) < 2:
        return None
    pick = rng.sample(pool, min(size, len(pool)))
    return Scenario(sid="neg_shuffle", units=[dict(u) for u in pick],
                    secret="", gt=frozenset())


def build_negatives(scen, rng=None, n_shuffle=None):
    """양성 목록 scen으로부터 음성 시나리오 목록 [(유형, Scenario), ...]를 만든다."""
    rng = rng or random.Random(0)
    negs = []
    for s in scen:
        d = distractor_only(s)
        if d:
            negs.append(("distractor", d))
    n_shuffle = n_shuffle if n_shuffle is not None else max(1, len(scen) // 2)
    for _ in range(n_shuffle):
        sh = shuffled(scen, rng, size=rng.randint(3, 5))
        if sh:
            negs.append(("shuffle", sh))
    return negs


if __name__ == "__main__":
    from datasets import load_patil
    pos = load_patil()
    negs = build_negatives(pos, rng=random.Random(0))
    from collections import Counter
    c = Counter(k for k, _ in negs)
    print(f"양성 {len(pos)}개 → 음성 {len(negs)}개 {dict(c)}")
    d = negs[0][1]
    print(f"  예(distractor): 단서 {len(d.units)}개, secret='' gt={sorted(d.gt)}")
