# -*- coding: utf-8 -*-
"""
exp3_mr_fidelity.py — small human audit of transformation fidelity for MR1 / MR3 / MR4.

The metamorphic checks assume that MR1 changes only wording, MR3 adds only unrelated sentences,
and MR4 changes only the identities of the people involved. This script draws N validated-path
excerpts from Experiment 3 (GPT-4o-mini paths by default), applies the three transformations
exactly as certify() does, and writes them side by side for two annotators.

  python3 exp3_mr_fidelity.py sheet  [--n 30] [--seed 0] [--log e2e_progress_openai_gpt-4o-mini-2024-07-18_r5.jsonl]
      -> results/mr_fidelity/fidelity_sheet.csv   (N x 3 rows: sid, mr, question, original, transformed, ann1, ann2, final)
  # annotators fill ann1 / ann2 with Y (property holds) or N (it does not); adjudicate in 'final'
  python3 exp3_mr_fidelity.py score
      -> per MR: fraction judged faithful, raw agreement and Cohen's kappa between annotators

No API calls; needs only the Patil data and spaCy.
"""
import argparse, csv, json, os, random, re
from datasets import load_patil, scenario_to_dialogue
from exp3_endtoend import render
import mr_nlp as MR

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "results", "logs")
ROOT = os.path.join(HERE, "..", "results", "mr_fidelity")
QUESTIONS = {
    "MR1": "Is the meaning preserved: same facts, people and values, only the wording changed?",
    "MR3": "Are the inserted sentences unrelated to the dialogue and to the secret (no cue added or removed)?",
    "MR4": "Were only the identities of people changed, consistently across the excerpt and the secret, with every other value intact?",
}


def cmd_sheet(a):
    log = os.path.join(LOGS, a.log)
    rows = [json.loads(l) for l in open(log, encoding="utf8") if l.strip()]
    with_path = [r for r in rows if r.get("S") and r.get("abc") != "C"]
    rng = random.Random(a.seed)
    picked = sorted(rng.sample(with_path, min(a.n, len(with_path))), key=lambda r: r["sid"])
    scen = {s.sid: s for s in load_patil()}
    out_rows = []
    for r in picked:
        s = scen[r["sid"]]; turns = scenario_to_dialogue(None, s, use_llm=False)
        S = sorted(r["S"]); ex = render(turns, S); z = s.secret
        # MR1
        out_rows.append(dict(sid=r["sid"], mr="MR1", question=QUESTIONS["MR1"], original=ex, transformed=MR.mr1_reframe(ex),
                             secret=z, ann1="", ann2="", final=""))
        # MR3
        noised, picked_sents = MR.mr3_inject_noise([turns[i] for i in S], 2)
        out_rows.append(dict(sid=r["sid"], mr="MR3", question=QUESTIONS["MR3"], original=ex, transformed=render(noised),
                             secret=z, ann1="", ann2="", final=""))
        # MR4 (excerpt and secret renamed with the same mapping, as in certify())
        ex4, mp4, _ = MR.mr4_rename(ex); z4 = z
        for old, new in sorted(mp4.items(), key=lambda kv: -len(kv[0])):
            z4 = re.sub(rf"\b{re.escape(old)}\b", new, z4)
        out_rows.append(dict(sid=r["sid"], mr="MR4", question=QUESTIONS["MR4"], original=ex,
                             transformed=ex4 + "\n\n[secret after renaming] " + z4 + "\n[mapping] " + json.dumps(mp4, ensure_ascii=False),
                             secret=z, ann1="", ann2="", final=""))
    os.makedirs(ROOT, exist_ok=True)
    out = os.path.join(ROOT, "fidelity_sheet.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"written: {out} ({len(picked)} scenarios x 3 MRs = {len(out_rows)} rows). Fill ann1/ann2 with Y or N; adjudicate in 'final'.")


def kappa(x, y):
    labels = sorted(set(x) | set(y)); n = len(x)
    po = sum(1 for u, v in zip(x, y) if u == v) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_score(a):
    sheet = os.path.join(ROOT, "fidelity_sheet.csv")
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    print(f"{'MR':<5}{'n':>4}{'faithful':>10}{'agree':>8}{'kappa':>8}")
    for mr in ("MR1", "MR3", "MR4"):
        rs = [r for r in rows if r["mr"] == mr]
        fin = [(r["final"] or r["ann1"]).strip().upper() for r in rs]
        a1 = [r["ann1"].strip().upper() for r in rs]; a2 = [r["ann2"].strip().upper() for r in rs]
        faithful = sum(1 for f in fin if f == "Y")
        agree = sum(1 for u, v in zip(a1, a2) if u == v) / len(rs) if all(a1) and all(a2) else float("nan")
        k = kappa(a1, a2) if all(a1) and all(a2) else float("nan")
        print(f"{mr:<5}{len(rs):>4}{faithful:>5}/{len(rs):<4}{agree:>8.3f}{k:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "score"])
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="e2e_progress_openai_gpt-4o-mini-2024-07-18_r5.jsonl")
    a = ap.parse_args()
    cmd_sheet(a) if a.cmd == "sheet" else cmd_score(a)


if __name__ == "__main__":
    main()
