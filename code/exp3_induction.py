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
    """Stratified semantic rate. Rows with an empty label are treated as not annotated. For each observer the
    annotated rows are split by the lexical criterion; the semantic rate is
        P(lex=1) * P(M | lex=1, annotated) + P(lex=0) * P(M | lex=0, annotated),
    which equals the plain rate when every row is annotated. The 95% CI is a stratified bootstrap (10,000)."""
    import random
    random.seed(a.seed)
    sheet = os.path.join(ROOT, "annotation_sheet.csv")
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    out = {}
    obs = sorted({r["observer"] for r in rows})
    print(f"{'observer':<45}{'n':>4}{'lex':>7}{'ann':>5}{'M|lex1':>9}{'M|lex0':>9}{'semantic':>10}{'95% CI':>16}{'agree':>7}")
    for o in obs:
        rs = [r for r in rows if r["observer"] == o]; n = len(rs)
        lab = {int(r["lexical"]): [] for r in rs}
        for r in rs:
            f = (r["final"] or r["ann1"]).strip().upper()
            if f:
                lab[int(r["lexical"])].append(f)
        p_lex = sum(int(r["lexical"]) for r in rs) / n
        def rate(l1, l0):
            m1 = l1.count("M") / len(l1) if l1 else 0.0; m0 = l0.count("M") / len(l0) if l0 else 0.0
            return p_lex * m1 + (1 - p_lex) * m0
        est = rate(lab[1], lab[0])
        bs = sorted(rate([random.choice(lab[1]) for _ in lab[1]], [random.choice(lab[0]) for _ in lab[0]]) for _ in range(10000))
        lo, hi = bs[249], bs[9750]
        ann = [(int(r["lexical"]), (r["final"] or r["ann1"]).strip().upper()) for r in rs if (r["final"] or r["ann1"]).strip()]
        agree = sum(1 for l, f in ann if l == (f == "M")) / len(ann) if ann else float("nan")
        notes = [r["note"] for r in rs if r.get("note")]
        res = {"n": n, "lexical": p_lex, "annotated": len(ann), "n_lex1_annotated": len(lab[1]), "n_lex0_annotated": len(lab[0]),
               "M_given_lex1": lab[1].count("M") / len(lab[1]) if lab[1] else None, "M_given_lex0": lab[0].count("M") / len(lab[0]) if lab[0] else None,
               "counts_lex1": {k: lab[1].count(k) for k in "MAX"}, "counts_lex0": {k: lab[0].count(k) for k in "MAX"},
               "semantic": est, "ci95": [lo, hi], "agreement_on_annotated": agree,
               "notes": {k: notes.count(k) for k in set(notes)}}
        out[o] = res
        print(f"{o:<45}{n:>4}{p_lex:>7.3f}{len(ann):>5}{(res['M_given_lex1'] or 0):>9.2f}{(res['M_given_lex0'] or 0):>9.2f}{est:>10.3f}   [{lo:.3f}, {hi:.3f}]{agree:>7.2f}")
    path = os.path.join(ROOT, "semantic_scores.json")
    json.dump(out, open(path, "w"), indent=1); print("written:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "sheet", "score"])
    ap.add_argument("--backend", default=None); ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--nli", action="store_true"); ap.add_argument("--seed", type=int, default=0)
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
