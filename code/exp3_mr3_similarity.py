# -*- coding: utf-8 -*-
"""
exp3_mr3_similarity.py — secondary, embedding-based check that MR3 insertions are unrelated to the excerpt.

mr_nlp.mr3_inject_noise selects sentences that share no content-word lemma with the dialogue. This script adds
a distributional check: for every returned path of Experiment 3 it recomputes the two inserted sentences and
reports the cosine similarity (averaged word vectors of spaCy en_core_web_md, stop words and punctuation
removed) between each inserted sentence and (a) every utterance of the excerpt and (b) the benchmark secret,
against the reference similarity between utterances of the same excerpt.

  pip install spacy && python -m spacy download en_core_web_md
  python3 exp3_mr3_similarity.py            # reads results/logs/e2e_progress_*_r5_mr4fix.jsonl (falls back to *_r5.jsonl)
No API calls.
"""
import glob, json, os, statistics as st
import numpy as np, spacy
from datasets import load_patil, scenario_to_dialogue
from exp3_endtoend import render
import mr_nlp as MR

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "logs")


def main():
    nlp = spacy.load("en_core_web_md")

    def vec(t):
        d = nlp(t); v = [tok.vector for tok in d if tok.has_vector and not tok.is_stop and not tok.is_punct]
        return np.mean(v, axis=0) if v else np.zeros(nlp.vocab.vectors.shape[1])

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    files = sorted(glob.glob(os.path.join(LOGS, "e2e_progress_*_r5_mr4fix.jsonl"))) or sorted(glob.glob(os.path.join(LOGS, "e2e_progress_*_r5.jsonl")))
    scen = {s.sid: s for s in load_patil()}
    seen = set(); maxsim, zsim, ref = [], [], []
    for f in files:
        for r in map(json.loads, open(f, encoding="utf8")):
            if not (r.get("S") and r.get("abc") != "C") or (r["sid"], tuple(r["S"])) in seen:
                continue
            seen.add((r["sid"], tuple(r["S"])))
            s = scen[r["sid"]]; turns = scenario_to_dialogue(None, s, use_llm=False); sel = [turns[i] for i in sorted(r["S"])]
            _, picked = MR.mr3_inject_noise(sel, 2)
            uv = [vec(t["text"]) for t in sel]; zv = vec(s.secret)
            for p in picked:
                pv = vec(p); maxsim.append(max(cos(pv, u) for u in uv)); zsim.append(cos(pv, zv))
            for i in range(len(uv)):
                for j in range(i + 1, len(uv)):
                    ref.append(cos(uv[i], uv[j]))
    print(f"distinct (scenario, path) excerpts: {len(seen)}; inserted sentences: {len(maxsim)}")
    print(f"max cosine between an inserted sentence and any utterance of its excerpt: mean {st.mean(maxsim):.3f}, "
          f"median {st.median(maxsim):.3f}, 95th percentile {np.percentile(maxsim, 95):.3f}, max {max(maxsim):.3f}")
    print(f"cosine between an inserted sentence and the benchmark secret: mean {st.mean(zsim):.3f}, max {max(zsim):.3f}")
    print(f"reference, cosine between utterances of the same excerpt: mean {st.mean(ref):.3f}, 5th percentile {np.percentile(ref, 5):.3f}")


if __name__ == "__main__":
    main()
