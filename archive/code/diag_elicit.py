#python3 diag_elicit.py --backend openrouter --model google/gemini-2.5-flash-lite --n 2# -*- coding: utf-8 -*-
"""diag_elicit.py — diagnoses the raw induction (elicitation) responses. Sends only the 3 framings for 1-2 scenarios (3-6 calls).
  python3 diag_elicit.py --backend openrouter --model google/gemini-2.5-flash-lite --n 2
When induction recall is abnormally low, it separates two causes: the model cannot draw out the secret (a real result),
or the response does not follow the JSON-list format and parsing fails (a format problem)."""
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
    print("=" * 70); print("sid", s.sid); print("benchmark secret:", s.secret[:160])
    for k, f in enumerate(ELICIT_FRAMINGS):
        raw = b.complete(JUDGE_SYS, f"대화:\n-----\n{ex}\n-----\n{f}")  # "대화:" = "Dialogue:" (prompt 3, verbatim)
        try:
            parsed = json.loads(raw); how = "json.loads OK"
        except Exception:
            parsed = re.findall(r'"([^"]+)"', raw); how = "json failed → quoted-string fallback"
        hit = mentions(parsed, s.secret) if parsed else False
        print(f"\n--- framing {k+1} · {how} · {len(parsed)} candidates · secret included {hit}")
        print("raw response (first 600 chars):", raw[:600].replace("\n", " ⏎ "))
