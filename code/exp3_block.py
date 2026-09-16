# -*- coding: utf-8 -*-
"""
================================================================================
 실험 3-3 — 차단(완화) 실험: 최소 차단 집합의 실효성 검증 (독립 실행, 실 LLM)
================================================================================
 리뷰어 지적 대응: 기여 3(최소 차단 집합)이 이론에 머문다는 문제를 실증으로 해소.
 절차(시나리오별):
   1) 인증 경로 수집: 탐색(find_path)→인증(certify). 발견 경로의 단서를 배제한
      재탐색 1회전으로 복수 경로 열거(P).
   2) 탐욕 hitting set: 가장 많은 경로에 걸치는 단서를 반복 선택 → 차단 집합 B.
   3) B의 발화를 "(content withheld)" 로 마스킹.
   4) 차단 후 재측정: (a) 직접 판정 V(마스킹 대화, z)  (b) 열린 유도로 z 재발견 여부.
 보고 지표:
   - 차단 집합 크기 |B| 평균, 대화 대비 비율 |B|/|U|
   - 차단 전 직접 판정 YES율(기준선) vs 차단 후 YES율(잔여 누출-판정)
   - 차단 후 유도 재발견율(잔여 누출-유도)
 실행(권장):
   OPENAI_API_KEY=.. python3 exp3_block.py --backend openai --model gpt-4o-mini \
       --limit 30 --r 5 --raw --sleep 1
================================================================================
"""
import argparse, random, re, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import QueryCounter
from exp3_endtoend import render, find_path, certify


def _secret_match(cands, secret):
    st = {w for w in re.findall(r"[a-z0-9]{3,}", secret.lower())}
    for c in cands:
        cw = {w for w in re.findall(r"[a-z0-9]{3,}", str(c).lower())}
        if len(st & cw) >= 2:
            return True
    return False


def enumerate_paths(obs, turns, z, max_paths=2):
    """인증 통과 경로를 열거한다: 기본 탐색 + (발견 경로 배제) 재탐색 1회전."""
    paths = []
    qc = QueryCounter()
    S1 = find_path(obs, turns, z, qc)
    if S1 is not None and len(S1) >= 2 and certify(obs, turns, S1, z):
        paths.append(set(S1))
        # 배제-재탐색: S1의 단서를 뺀 나머지에서 두 번째 경로 시도
        keep = [i for i in range(len(turns)) if i not in S1]
        if len(keep) >= 2:
            sub = [turns[i] for i in keep]
            qc2 = QueryCounter()
            S2r = find_path(obs, sub, z, qc2)
            if S2r is not None and len(S2r) >= 2:
                S2 = {keep[i] for i in S2r}  # 원 인덱스로 환원
                sub_sel = [turns[i] for i in sorted(S2)]
                if certify(obs, turns, S2, z):
                    paths.append(S2)
    return paths[:max_paths]


def greedy_hitting_set(paths):
    """탐욕 근사: 매 단계 가장 많은 미타격 경로에 걸치는 단서 선택."""
    B, remaining = set(), [set(p) for p in paths]
    while remaining:
        # 후보 단서별 커버 수
        from collections import Counter
        cnt = Counter()
        for p in remaining:
            for u in p:
                cnt[u] += 1
        best = max(cnt, key=lambda u: cnt[u])
        B.add(best)
        remaining = [p for p in remaining if best not in p]
    return B


def mask(turns, B):
    return [{"speaker": t["speaker"],
             "text": "(content withheld)" if i in B else t["text"]}
            for i, t in enumerate(turns)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--raw", action="store_true", help="값 그대로 렌더(3-1과 동일, 권장)")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model)
    scen = load_patil()
    scen = scen[args.start:args.start + args.limit] if args.limit else scen[args.start:]
    print(f"실험 3-3 차단 — {len(scen)}개 시나리오 (backend={args.backend}, r={args.r})")

    n_cert = 0            # 인증 경로가 1개 이상 나온 시나리오 수
    sizes, ratios = [], []
    pre_yes = post_yes = post_elicit = 0
    multi = 0             # 경로 2개 이상 열거된 시나리오

    done = 0
    for k, s in enumerate(scen):
        try:
            obs = Observer(backend, r=args.r)
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            z = s.secret

            paths = enumerate_paths(obs, turns, z)
            done += 1
            if not paths:
                if (k + 1) % 5 == 0: print(f"  ...{k+1}/{len(scen)}")
                if args.sleep: time.sleep(args.sleep)
                continue
            n_cert += 1
            if len(paths) >= 2: multi += 1

            B = greedy_hitting_set(paths)
            sizes.append(len(B)); ratios.append(len(B) / len(turns))

            # 기준선: 차단 전 직접 판정 (인증 경로가 있으므로 대개 YES)
            pre_yes += int(obs.judge_excerpt(render(turns), z) == 1)
            # 차단 후: 직접 판정 + 열린 유도 재발견
            blocked = mask(turns, B)
            post_yes += int(obs.judge_excerpt(render(blocked), z) == 1)
            cands = obs.elicit(render(blocked))
            post_elicit += int(_secret_match(cands, z))

            if (k + 1) % 5 == 0: print(f"  ...{k+1}/{len(scen)}")
            if args.sleep: time.sleep(args.sleep)
        except Exception as e:
            print(f"  [중단] {k+1}번째에서 한도/오류: {str(e)[:120]}")
            print(f"  → 여기까지 {done}개 완료분으로 집계합니다.")
            break

    print("\n" + "=" * 62 + "\n실험 3-3 차단(완화) 결과\n" + "=" * 62)
    print(f"인증 경로 확보 시나리오: {n_cert}/{done} (경로 2개 이상: {multi})")
    if sizes:
        import statistics as st
        print(f"차단 집합 크기 |B| 평균 {st.mean(sizes):.2f} · 대화 대비 비율 {st.mean(ratios):.1%}")
        print(f"직접 판정 YES율  차단 전 {pre_yes}/{n_cert} → 차단 후 {post_yes}/{n_cert}"
              f"  (잔여 누출-판정 {post_yes/n_cert:.3f})")
        print(f"차단 후 유도 재발견율 {post_elicit}/{n_cert} = {post_elicit/n_cert:.3f}  (잔여 누출-유도)")
        print("해석: 차단 후 두 잔여율이 낮을수록 표적 완화가 실효적. 남는 값은")
        print("      사전지식 기반 재추론의 실측치로서 §VI 한계 논의와 연결해 보고.")


if __name__ == "__main__":
    main()
