# -*- coding: utf-8 -*-
"""
exp3_induction.py — secret induction re-scored semantically (reviewer item 6).

Stage 1 only: each observer is shown the whole dialogue (target secret withheld) and asked, with the
same three open-ended framings as Experiment 3, what sensitive conclusions it draws. The induced
candidates are saved verbatim so that they can be scored (a) by the lexical criterion of the paper,
(b) by two human annotators, and optionally (c) by an NLI model.

Steps
  1. dump   — query the observer and write results/induction/<tag>.jsonl (one line per scenario:
              sid, benchmark secret z, candidate list, lexical match)
  2. sheet  — write results/induction/annotation_sheet.csv: one row per (scenario, observer) with
              z and the candidates; annotators fill Match / Mismatch / Ambiguous in their column
  3. score  — read the filled sheet and print, per observer: lexical rate, semantic rate
              (Match by both annotators after adjudication), agreement between criteria, Cohen's kappa
              between annotators; optionally --nli adds an entailment-based rate (needs transformers)

  export OPENAI_API_KEY=...; export OPENROUTER_API_KEY=...        # never commit keys
  python3 exp3_induction.py dump  --backend openai     --model gpt-4o-mini-2024-07-18   --raw
  python3 exp3_induction.py dump  --backend openrouter --model qwen/qwen3.6-27b          --raw
  python3 exp3_induction.py dump  --backend openrouter --model google/gemini-2.5-flash-lite --raw
  python3 exp3_induction.py sheet
  # ... two annotators fill columns ann1 / ann2 (M = Match, X = Mismatch, A = Ambiguous), adjudicate in 'final'
  python3 exp3_induction.py score [--nli]
Cost: 3 open-ended calls per scenario per observer (348 x 3 calls in total).
"""
import argparse, csv, glob, json, os, sys
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from exp3_endtoend import render, mentions

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "induction")


def tag_of(a):
    return f"{a.backend}_{(a.model or 'default').replace('/', '_')}"


def cmd_dump(a):
    os.makedirs(ROOT, exist_ok=True)
    out = os.path.join(ROOT, f"{tag_of(a)}.jsonl")
    done = {json.loads(l)["sid"] for l in open(out, encoding="utf8")} if os.path.exists(out) else set()
    scen = load_patil()[:a.limit] if a.limit else load_patil()
    if not scen:
        sys.exit("no Patil scenarios found under data/patil — run `python3 fetch_patil.py` first")
    todo = [s for s in scen if s.sid not in done]
    print(f"{tag_of(a)}: {len(scen)} scenarios, {len(done)} already done, {len(todo)} to query")
    backend = make_backend(a.backend, a.model); obs = Observer(backend, r=1)
    for s in scen:
        if s.sid in done: continue
        turns = scenario_to_dialogue(backend, s, use_llm=not a.raw)
        cands = [str(c) for c in obs.elicit(render(turns))]
        rec = {"sid": s.sid, "observer": tag_of(a), "z": s.secret, "candidates": cands, "lexical": bool(mentions(cands, s.secret))}
        with open(out, "a", encoding="utf8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("written:", out)


def cmd_sheet(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*.jsonl"))):
        for line in open(f, encoding="utf8"):
            r = json.loads(line)
            rows.append({"sid": r["sid"], "observer": r["observer"], "benchmark_secret": r["z"],
                         "candidates": " || ".join(r["candidates"]), "lexical": int(r["lexical"]), "ann1": "", "ann2": "", "final": ""})
    if not rows:
        sys.exit(f"no induction results found in {os.path.abspath(ROOT)} — run the dump step first")
    rows.sort(key=lambda r: (r["sid"], r["observer"]))
    out = os.path.join(ROOT, "annotation_sheet.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"written: {out} ({len(rows)} rows). Fill ann1/ann2 with M (Match), X (Mismatch) or A (Ambiguous); put the adjudicated label in 'final'.")


def kappa(x, y):
    labels = sorted(set(x) | set(y)); n = len(x)
    po = sum(1 for u, v in zip(x, y) if u == v) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_score(a):
    sheet = os.path.join(ROOT, "annotation_sheet.csv")
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    nli = None
    if a.nli:
        from transformers import pipeline
        nli = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    obs = sorted({r["observer"] for r in rows})
    print(f"{'observer':<45}{'n':>5}{'lexical':>9}{'semantic':>10}{'agree':>8}{'kappa':>8}" + ("{:>8}".format("nli") if nli else ""))
    for o in obs:
        rs = [r for r in rows if r["observer"] == o]
        lex = [int(r["lexical"]) for r in rs]
        fin = [(r["final"] or r["ann1"]).strip().upper() for r in rs]
        sem = [1 if f == "M" else 0 for f in fin]
        agree = sum(1 for l, s in zip(lex, sem) if l == s) / len(rs)
        a1 = [r["ann1"].strip().upper() for r in rs]; a2 = [r["ann2"].strip().upper() for r in rs]
        k = kappa(a1, a2) if all(a1) and all(a2) else float("nan")
        line = f"{o:<45}{len(rs):>5}{sum(lex)/len(rs):>9.3f}{sum(sem)/len(rs):>10.3f}{agree:>8.3f}{k:>8.3f}"
        if nli:
            hits = 0
            for r in rs:
                ok = False
                for c in r["candidates"].split(" || "):
                    f1 = nli({"text": c, "text_pair": r["benchmark_secret"]})[0]; f2 = nli({"text": r["benchmark_secret"], "text_pair": c})[0]
                    if f1["label"].lower() == "entailment" and f2["label"].lower() == "entailment":
                        ok = True; break
                hits += int(ok)
            line += f"{hits/len(rs):>8.3f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "sheet", "score"])
    ap.add_argument("--backend", default=None); ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--nli", action="store_true")
    a = ap.parse_args()
    if a.cmd == "dump":
        assert a.backend, "--backend is required for dump"
        cmd_dump(a)
    elif a.cmd == "sheet":
        cmd_sheet(a)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
