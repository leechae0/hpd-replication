# -*- coding: utf-8 -*-
"""
export_dialogues.py — Writes the 116 natural-language dialogues exactly as presented to the observers
in Experiment 3 (value-preserving rendering, i.e. exp3_endtoend.py --raw) to
data/dialogues/patil_dialogues_116.jsonl.

Each line: {"sid", "secret", "gt" (indices of the ground-truth cue set), "turns": [{"speaker","text","unit_idx"}]}
The rendering is deterministic (no LLM call), so the file can be regenerated at any time.

    python3 export_dialogues.py
"""
import json, os
from datasets import load_patil, scenario_to_dialogue

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "dialogues", "patil_dialogues_116.jsonl")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    scen = load_patil()
    with open(OUT, "w", encoding="utf-8") as f:
        for s in scen:
            turns = scenario_to_dialogue(None, s, use_llm=False)
            f.write(json.dumps({"sid": s.sid, "secret": s.secret, "gt": sorted(s.gt),
                                "turns": turns}, ensure_ascii=False) + "\n")
    print(f"{len(scen)} dialogues written to {os.path.abspath(OUT)}")
