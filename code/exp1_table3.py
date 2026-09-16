# -*- coding: utf-8 -*-
"""
exp1_table3.py — Reproduces Table III of the paper (Experiment 1) exactly as reported.

Same oracle, noise model, seeds and search as exp1_search.py. The only difference from
exp1_search.py is the MuSiQue sample: the paper uses a hop-stratified sample of the
MuSiQue-Answerable validation split — the first 500 2-hop, 500 3-hop and 405 4-hop
answerable questions in file order (1,405 in total) — instead of the first 500 questions.

  python3 exp1_table3.py --hotpotqa data/hotpot_dev_distractor_v1.json \
                         --musique  data/musique_ans_v1.0_dev.jsonl

Prints, per dataset: exact recovery (mean ± sd over 5 seeds) for the error-free and
(ε=0.20, r=9) conditions, mean Jaccard, HPD query count (|U|+1) and the dataset-average
exhaustive query count 2^|U|.
"""
import argparse, statistics as st
from datasets import load_patil, load_hotpotqa, load_musique
from exp1_search import run_dataset


def load_musique_stratified(path, quota=((2, 500), (3, 500), (4, 405))):
    out = []
    for hops, n in quota:
        out += load_musique(path, min_hops=hops, max_hops=hops, limit=n)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patil", default=None)
    ap.add_argument("--hotpotqa", default=None)
    ap.add_argument("--musique", default=None)
    a = ap.parse_args()
    run_dataset("Patil (8 cues)", load_patil(a.patil))
    if a.hotpotqa:
        run_dataset("HotpotQA (10 cues)", load_hotpotqa(a.hotpotqa))
    if a.musique:
        scen = load_musique_stratified(a.musique)
        from collections import Counter
        print("MuSiQue stratified sample:", len(scen), "questions; hops:",
              dict(Counter(len(s.gt) for s in scen)), "; cue counts:",
              dict(Counter(len(s.units) for s in scen)))
        run_dataset("MuSiQue (20 cues, 2-4 hop stratified)", scen)
