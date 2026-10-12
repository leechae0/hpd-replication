#python3 diag_elicit.py --backend openrouter --model google/gemini-2.5-flash-lite --n 2# -*- coding: utf-8 -*-
"""diag_elicit.py — 유도(elicitation) 응답 원문 진단. 시나리오 1~2개에 프레이밍 3개만 보낸다(호출 3~6회).
  python3 diag_elicit.py --backend openrouter --model google/gemini-2.5-flash-lite --n 2
유도 재현율이 비정상적으로 낮을 때, 모델이 비밀을 못 끌어내는 것인지(진짜 결과)
JSON 리스트 형식을 안 지켜 파싱이 실패하는 것인지(형식 문제)를 가른다."""
import argparse, json, re
from llm import make_backend, JUDGE_SYS, ELICIT_FRAMINGS
from datasets import load_patil, scenario_to_dialogue
from exp3_endtoend import render, mentions

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--n", type=int, default=2)
a = ap.parse_args()
b = make_backend(a.backend, a.model)
for s in load_patil()[:a.n]:
    turns = scenario_to_dialogue(b, s, use_llm=False)
    ex = render(turns)
    print("=" * 70); print("sid", s.sid); print("정답 비밀:", s.secret[:160])
    for k, f in enumerate(ELICIT_FRAMINGS):
        raw = b.complete(JUDGE_SYS, f"대화:\n-----\n{ex}\n-----\n{f}")
        try:
            parsed = json.loads(raw); how = "json.loads OK"
        except Exception:
            parsed = re.findall(r'"([^"]+)"', raw); how = "json 실패 → 따옴표 폴백"
        hit = mentions(parsed, s.secret) if parsed else False
        print(f"\n--- 프레이밍 {k+1} · {how} · 후보 {len(parsed)}개 · 정답 포함 {hit}")
        print("원문(앞 600자):", raw[:600].replace("\n", " ⏎ "))
