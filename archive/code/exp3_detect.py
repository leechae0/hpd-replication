# -*- coding: utf-8 -*-
"""
================================================================================
 Experiment 3-2 — dialogue-level leakage detection (stand-alone, needs a real LLM)
================================================================================
 From a pool mixing positives (Patil dialogues with a secret) and negatives (three kinds of dialogue without one),
 the full induction → search → validation pipeline is applied with labels hidden to pick out the leaking dialogues.
   Decision: a dialogue is flagged as 'leaking' if any path passes validation.
   Metrics: dialogue-level precision, recall, F1, false-positive rate; for true positives, secret match and path Jaccard.
 Negatives are built by negatives.py (sensitive combination removed / cues shuffled / natural benign dialogues = MAGPIE, optional).
 Usage:
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
    """A match if any induction candidate overlaps the core content of the benchmark secret.
    (Measured with a fixed threshold of 2 shared content words instead of a strict threshold proportional to the label length.)"""
    import re as _re
    st = {w for w in _re.findall(r"[a-z0-9]{3,}", secret.lower())}
    for c in cands:
        cw = {w for w in _re.findall(r"[a-z0-9]{3,}", str(c).lower())}
        if len(st & cw) >= 2:
            return True
    return False

def detect(obs, turns, max_cands=5):
    """Draw secret candidates by induction and apply search + validation to each.
    Returns (True, secret, path) if some path passes validation, otherwise (False, None, None)."""
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
    ap.add_argument("--limit", type=int, default=20, help="maximum number of positive (Patil) scenarios")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--magpie", default=None, help="path to natural benign dialogues (MAGPIE), optional")
    ap.add_argument("--raw", action="store_true", help="render the dialogue with values as is (same as 3-1; recommended)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between scenarios (eases rate limits)")
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
    print(f"detection pool: {len(pos)} positives · {len(negs)} negatives "
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
            print(f"  [stopped] error at item {i+1}: {str(e)[:100]}")
            print(f"  → aggregating the results of the completed items ({TP+FP+FN+TN}).")
            break

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = FP / (FP + TN) if FP + TN else 0.0

    print("\n" + "=" * 62 + "\nExperiment 3-2 dialogue-level detection results\n" + "=" * 62)
    print(f"confusion matrix  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"precision {prec:.3f} · recall {rec:.3f} · F1 {f1:.3f} · false-positive rate {fpr:.3f}")
    if sec_match:
        print(f"for true positives  secret match {st.mean(sec_match):.3f} · path Jaccard {st.mean(path_jac):.3f}")
    if by_kind:
        print("false positives by negative type: " +
              ", ".join(f"{k} {by_kind_fp[k]}/{by_kind[k]}" for k in by_kind))


if __name__ == "__main__":
    main()
