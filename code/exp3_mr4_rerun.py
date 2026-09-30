# -*- coding: utf-8 -*-
"""
exp3_mr4_rerun.py — re-judge the MR4 (entity substitution) check of Experiment 3 with the fixed
name filter in mr_nlp.mr4_rename (2026-09-21).

Why: the original MR4 renamed every span that spaCy tagged as PERSON, which in record-style
dialogues included key=value fields ("employee_id=201"), street names ("Main St") and capitalized
values ("Chronic Asthma"). Renaming those corrupts the data instead of changing identities only,
so some MR4 failures in the 2026-08/09 runs were transformation artifacts. The fixed filter renames
only speakers, values of person-name fields, and words of those names.

What it does: for every scenario in results/logs/e2e_progress_<tag>_r5.jsonl that has a returned
path S, it rebuilds the excerpt, applies the fixed MR4, asks the same observer the same judgment
(r votes, majority), and writes results/logs/e2e_progress_<tag>_r5_mr4fix.jsonl — a copy of the
original log in which cd.MR4, cert and cert4 are updated. Records without a path are copied as is.
Each rewritten record also keeps MR4_old (the old verdict) and mr4_old_nonname (True if the old
transformation had renamed at least one non-name token in this excerpt). Resumable.

  export OPENAI_API_KEY=...; export OPENROUTER_API_KEY=...       # never commit keys
  python3 exp3_mr4_rerun.py --backend openai     --model gpt-4o-mini-2024-07-18   --r 5 --raw
  python3 exp3_mr4_rerun.py --backend openrouter --model qwen/qwen3.6-27b          --r 5 --raw
  python3 exp3_mr4_rerun.py --backend openrouter --model google/gemini-2.5-flash-lite --r 5 --raw
  python3 exp3_mr4_rerun.py --backend openai --model gpt-4o-mini-2024-07-18 --report-only
Cost: one MR4 judgment (r calls) per returned path, about 250 x 5 calls over the three observers.
Afterwards: python3 exp3_stats.py --suffix _mr4fix   and   python3 exp3_compare.py --glob "e2e_progress_*_mr4fix.jsonl"
"""
import argparse, json, os, re, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from exp3_endtoend import render, _prog_path
import mr_nlp as MR

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "logs")


def old_candidates(text):
    """Name candidates of the unfiltered MR4 (every PERSON span plus speakers)."""
    doc = MR._NLP(text)
    return set(MR._person_spans(doc).keys()) | MR._speaker_names(text)


def report(rows):
    paths = [r for r in rows if r.get("S") and "cd" in r and r.get("abc") != "C"]
    n = len(paths)
    old4 = sum(1 for r in paths if r.get("MR4_old", r["cd"]["MR4"])); new4 = sum(1 for r in paths if r["cd"]["MR4"])
    oldc = sum(1 for r in paths if r.get("cert_old", r["cert"])); newc = sum(1 for r in paths if r["cert"])
    aff = [r for r in paths if r.get("mr4_old_nonname")]
    print(f"paths (A/B with a returned path): {n}")
    print(f"MR4 pass: old {old4}/{n} = {old4/n:.3f}  ->  new {new4}/{n} = {new4/n:.3f}")
    print(f"final 5/5: old {oldc}/{n} = {oldc/n:.3f}  ->  new {newc}/{n} = {newc/n:.3f}")
    if aff:
        a_old = sum(1 for r in aff if r.get("MR4_old")); a_new = sum(1 for r in aff if r["cd"]["MR4"])
        print(f"excerpts where the old MR4 renamed a non-name token: {len(aff)}/{n}; MR4 pass there old {a_old} -> new {a_new}")
    flips = sum(1 for r in paths if r.get("MR4_old") is not None and r["MR4_old"] != r["cd"]["MR4"])
    print(f"MR4 verdict changed in {flips} paths")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--raw", action="store_true", help="render dialogues from the raw values (as in the paper runs)")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    src = os.path.join(args.logs, os.path.basename(_prog_path(args)))
    assert os.path.exists(src), f"source log not found: {src}"
    out = src[:-len(".jsonl")] + "_mr4fix.jsonl"
    done = {}
    if os.path.exists(out):
        for line in open(out, encoding="utf8"):
            r = json.loads(line); done[r["sid"]] = r
    src_rows = [json.loads(l) for l in open(src, encoding="utf8") if l.strip()]
    if args.report_only:
        report([done.get(r["sid"], r) for r in src_rows]); return
    scen = {s.sid: s for s in load_patil()}
    todo = [r for r in src_rows if r["sid"] not in done]
    print(f"{os.path.basename(src)}: {len(src_rows)} records, {len(done)} already re-judged, {len(todo)} to do")
    backend = make_backend(args.backend, args.model); obs = Observer(backend, r=args.r)
    for r in todo:
        rec = dict(r)
        if r.get("S") and "cd" in r:
            s = scen[r["sid"]]
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            ex = render(turns, r["S"]); z = s.secret
            ex4, mp4, _ = MR.mr4_rename(ex)
            z4 = z
            for old, new in sorted(mp4.items(), key=lambda kv: -len(kv[0])):
                z4 = re.sub(rf"\b{re.escape(old)}\b", new, z4)
            r4 = obs.judge_excerpt(ex4, z4) == 1
            cd = dict(r["cd"]); rec["MR4_old"] = bool(cd["MR4"]); cd["MR4"] = bool(r4)
            rec["cd"] = cd; rec["cert_old"] = bool(r["cert"])
            rec["cert"] = bool(cd["base"] and cd["single"] and cd["MR1"] and cd["MR3"] and cd["MR4"] and len(r["S"]) >= 2)
            rec["cert4"] = sum(int(cd[k]) for k in ("base", "single", "MR1", "MR3", "MR4")) >= 4
            rec["mr4_old_nonname"] = bool(old_candidates(ex) - set(mp4))
            rec["mr4_names"] = mp4
        with open(out, "a", encoding="utf8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[r["sid"]] = rec
        if args.sleep: time.sleep(args.sleep)
    print("written:", out)
    report([done.get(r["sid"], r) for r in src_rows])


if __name__ == "__main__":
    main()
