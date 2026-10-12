# -*- coding: utf-8 -*-
"""
================================================================================
 실험 3-2 — 대화 수준 누출 탐지 (독립 실행, 실 LLM 필요)
================================================================================
 양성(비밀 있는 Patil 대화) + 음성(비밀 없는 대화 3종)을 섞은 풀에서,
 레이블을 감춘 채 유도→탐색→인증 전 과정을 적용해 누출 대화를 골라낸다.
   판정: 인증을 통과한 경로가 하나라도 있으면 '누출'로 플래그.
   지표: 대화 수준 정밀도·재현율·F1·오탐률, 양성 적중 시 기밀 일치도·경로 자카드.
 음성은 negatives.py로 구성(민감 조합만 제거 / 단서 뒤섞기 / 자연 정상=MAGPIE 선택).
 실행:
   GROQ_API_KEY=..  python3 exp3_detect.py --backend groq --limit 20 --r 3
   OPENAI_API_KEY=.. python3 exp3_detect.py --backend openai --model gpt-4o-2024-08-06
================================================================================
"""
import argparse, random, statistics as st, time
from collections import Counter
from llm import make_backend, Observer
from datasets import load_patil, load_magpie, scenario_to_dialogue
from search import jaccard, QueryCounter
from exp3_endtoend import render, find_path, certify, mentions
import negatives as NEG


def _secret_match(cands, secret):
    """유도 후보 집합 중 하나라도 정답 비밀의 핵심 내용과 겹치면 일치로 본다.
    (라벨 길이에 비례하는 엄격 임계 대신, 고정 임계 2개 내용어 겹침으로 공정하게 측정.)"""
    import re as _re
    st = {w for w in _re.findall(r"[a-z0-9]{3,}", secret.lower())}
    for c in cands:
        cw = {w for w in _re.findall(r"[a-z0-9]{3,}", str(c).lower())}
        if len(st & cw) >= 2:
            return True
    return False

def detect(obs, turns, max_cands=5):
    """유도로 비밀 후보를 뽑고, 각 후보에 탐색+인증을 적용.
    인증을 통과한 경로가 있으면 (True, 비밀, 경로), 없으면 (False, None, None)."""
    full = render(turns)
    cands = []
    for z in obs.elicit(full):
        z = str(z).strip()
        if z and z not in cands:
            cands.append(z)
        if len(cands) >= max_cands:
            break
    for z in cands:
        qc = QueryCounter()
        S = find_path(obs, turns, z, qc)
        if S is not None and len(S) >= 2 and certify(obs, turns, S, z):
            return True, z, S, cands
    return False, None, None, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=20, help="양성(Patil) 시나리오 수 상한")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--magpie", default=None, help="자연 정상 대화(MAGPIE) 경로(선택)")
    ap.add_argument("--raw", action="store_true", help="대화를 값 그대로 렌더(3-1과 동일, 권장)")
    ap.add_argument("--sleep", type=float, default=0.0, help="시나리오 간 대기 초(rate limit 완화)")
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model)
    obs = Observer(backend, r=args.r)
    rng = random.Random(0)

    pos = load_patil()
    if args.limit:
        pos = pos[:args.limit]
    negs = NEG.build_negatives(pos, rng=rng, n_shuffle=max(1, len(pos) // 2))
    if args.magpie:
        for s in load_magpie(args.magpie, limit=max(1, len(pos) // 2)):
            negs.append(("natural", s))

    pool = [("pos", "leak", s) for s in pos] + [("neg", k, s) for k, s in negs]
    rng.shuffle(pool)
    print(f"탐지 풀: 양성 {len(pos)} · 음성 {len(negs)} "
          f"(backend={args.backend}, r={args.r})")

    TP = FP = TN = FN = 0
    sec_match, path_jac = [], []
    by_kind, by_kind_fp = Counter(), Counter()

    for i, (label, kind, s) in enumerate(pool):
        try:
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            flagged, z, S, cands = detect(obs, turns)
            if label == "pos":
                if flagged:
                    TP += 1
                    sec_match.append(int(_secret_match(cands, s.secret)))
                    path_jac.append(jaccard(S, s.gt))
                else:
                    FN += 1
            else:
                by_kind[kind] += 1
                if flagged:
                    FP += 1; by_kind_fp[kind] += 1
                else:
                    TN += 1
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{len(pool)}")
            if args.sleep:
                time.sleep(args.sleep)
        except Exception as e:
            print(f"  [중단] {i+1}번째에서 오류로 멈춤: {str(e)[:100]}")
            print(f"  → 여기까지 완료분({TP+FP+FN+TN}개)으로 결과를 집계합니다.")
            break

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = FP / (FP + TN) if FP + TN else 0.0

    print("\n" + "=" * 62 + "\n실험 3-2 대화 수준 탐지 결과\n" + "=" * 62)
    print(f"혼동행렬  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"정밀도 {prec:.3f} · 재현율 {rec:.3f} · F1 {f1:.3f} · 오탐률 {fpr:.3f}")
    if sec_match:
        print(f"양성 적중 시  기밀 일치도 {st.mean(sec_match):.3f} · 경로 자카드 {st.mean(path_jac):.3f}")
    if by_kind:
        print("음성 유형별 오탐: " +
              ", ".join(f"{k} {by_kind_fp[k]}/{by_kind[k]}" for k in by_kind))


if __name__ == "__main__":
    main()
