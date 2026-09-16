# -*- coding: utf-8 -*-
"""
================================================================================
 실험 2 — 메타모픽 인증의 타당성 (독립 실행)
================================================================================
 MR 변형은 실제 자연어 처리 코드(mr_nlp: spaCy NER·상호참조·검색)로 수행한다. mock 아님.
 통제 실험 설계상, 진성/스퓨리어스 '판정 거동'을 주입한 관찰자를 사용한다.
   - 진성 관찰자: 정답 단서가 모두 언급되면 YES. 재표현·개명·무관주입에 불변.
   - 스퓨리어스 관찰자 4종: 각기 한 실패 유형(표면 표현·개체명·무관 불안정·우연 상관)에 취약.
 (주의: 여기서 '주입'은 관찰자의 거동을 통제하기 위한 실험 장치이며, 변형 자체는 실 NLP 코드다.)

 측정:
   [0] 구현 검증 : 각 변형이 명세대로 수행되는가 (결정적 검사)
   [1] soundness : 진성 경로가 인증을 통과하는가 (≈ recall)
   [2] coverage  : 스퓨리어스 경로가 기각되는가 (≈ precision), 독립 유형 풀
   [3] 비중복성   : 검사 하나를 제거하면 그 유형이 통과하는가 (ablation)
 실행: python3 exp2_certify.py   (spaCy en_core_web_sm 필요)
================================================================================
"""
import re, random, statistics as st
from collections import Counter
from datasets import load_patil, scenario_to_dialogue  # scenario_to_dialogue는 실 LLM 변환용(선택)
import mr_nlp as MR


# --- 통제 관찰자: 자연어 발화 기반. 정답 테이블 명이 모두 언급되면 YES. ---
def _phrase(u):  # 단서를 식별하는 명사구(대화 문장에 등장)
    return u["key"]

class ControlledObserver:
    """진성(brittle=None) 또는 스퓨리어스(brittle 지정) 거동을 갖는 통제 관찰자."""
    def __init__(self, scenario, brittle=None):
        self.keys = {scenario.units[i]["key"] for i in scenario.gt}
        self.names = set(scenario.entities)
        self.brittle = brittle

    def judge_excerpt(self, excerpt, z=None):
        low = excerpt.lower()
        if not all(k.lower() in low for k in self.keys):
            return 0
        # 표면 표현 의존: MR1 재표현이 'earlier today'(→'before this morning')를 바꾸면 무너짐
        # (datasets.py의 현재 발화 렌더링 "... was reviewing the '<table>' records earlier today."에 맞춤)
        if self.brittle == "surface" and "earlier today" not in low:
            return 0
        # 개체명 의존: MR4 개명으로 원래 이름이 사라지면 무너짐
        if self.brittle == "entity" and not any(n.lower() in low for n in self.names):
            return 0
        # 무관 정보에 불안정: MR3가 'records'가 없는 무관 발화를 넣으면 무너짐
        if self.brittle == "noise" and any(
                ln.strip() and "records" not in ln for ln in low.splitlines()):
            return 0
        # cooccur(우연 상관): 어떤 변형에도 불변 → 어떤 MR에도 걸리지 않음(정직한 구멍)
        return 1


# 대화를 단서 키가 텍스트에 드러나도록 렌더 (통제 실험용)
def render(units, idx=None):
    sel = units if idx is None else [units[i] for i in sorted(idx)]
    return "\n".join(f'{u.get("agent") or u.get("speaker") or "?"}: {u["text"]}' for u in sel)


def certify(observer, units, S):
    ex = render(units, S)
    base = observer.judge_excerpt(ex) == 1
    single = all(observer.judge_excerpt(render(units, [i])) == 0 for i in S) if len(S) >= 2 else False
    r1 = observer.judge_excerpt(MR.mr1_reframe(ex)) == 1
    noised, _ = MR.mr3_inject_noise([units[i] for i in sorted(S)], 2)
    r3 = observer.judge_excerpt(render(noised)) == 1
    ex4, _mp, _ = MR.mr4_rename(ex)
    r4 = observer.judge_excerpt(ex4) == 1
    return {"single": single, "MR1": r1, "MR3": r3, "MR4": r4,
            "certified": base and single and r1 and r3 and r4 and len(S) >= 2}

def certify_drop(observer, units, S, drop):
    r = certify(observer, units, S)
    keep = {k: (True if k == drop else r[k]) for k in ("single", "MR1", "MR3", "MR4")}
    return all(keep.values()) and observer.judge_excerpt(render(units, S)) == 1 and len(S) >= 2


# --- [측정 0] 변형 구현 검증 (실 NLP, 결정적) ---
def measure0(scen):
    agg = Counter(); tot = Counter()
    for s in scen:
        ex = render(s.units, s.gt)
        v1 = MR.verify_mr1(ex, MR.mr1_reframe(ex))
        tot["MR1_인물보존"] += 1; agg["MR1_인물보존"] += v1["인물_보존"]
        r4, mp, _ = MR.mr4_rename(ex)
        v4 = MR.verify_mr4(ex, r4, mp)
        tot["MR4_원래이름_0"] += 1; agg["MR4_원래이름_0"] += v4["원래이름_0개남음"]
        noised, picked = MR.mr3_inject_noise(s.units, 2)
        v3 = MR.verify_mr3(s.units, picked)
        tot["MR3_추가_무관"] += 1; agg["MR3_추가_무관"] += v3["추가문장_모두_무관"]
    return {k: (agg[k], tot[k]) for k in tot}


if __name__ == "__main__":
    scen = load_patil()
    print(f"실험 2 (실 spaCy MR) — Patil {len(scen)}개\n")

    print("[측정 0] 구현 검증 (변형이 명세대로 — 결정적)")
    for k, (ok, t) in measure0(scen).items():
        print(f"   {'OK ' if ok==t else 'CHK'} {k:<16} {ok}/{t}")

    ok = sum(certify(ControlledObserver(s), s.units, set(s.gt))["certified"] for s in scen)
    print(f"\n[측정 1] soundness (진성 유지 ≈ recall): {ok}/{len(scen)} = {ok/len(scen):.3f}")

    kinds = ["surface", "entity", "noise", "cooccur"]
    caught, tot = Counter(), Counter()
    for s in scen:
        for k in kinds:
            tot[k] += 1
            if not certify(ControlledObserver(s, brittle=k), s.units, set(s.gt))["certified"]:
                caught[k] += 1
    cov = sum(caught.values()) / sum(tot.values())
    print(f"\n[측정 2] coverage (스퓨리어스 검출 ≈ precision): {cov:.1%}")
    for k in kinds:
        note = "  ← 잔여 오류(우연 상관, 정직 보고)" if k == "cooccur" and caught[k] == 0 else ""
        print(f"   {k:<8} {caught[k]}/{tot[k]}{note}")

    print(f"\n[측정 3] 비중복성 (검사 제거 시 해당 유형 통과):")
    for k, mr in {"surface": "MR1", "entity": "MR4", "noise": "MR3"}.items():
        leak = sum(1 for s in scen if certify_drop(ControlledObserver(s, brittle=k), s.units, set(s.gt), mr))
        print(f"   {mr} 제거 → {k} 유형 통과: {leak}/{len(scen)}")
