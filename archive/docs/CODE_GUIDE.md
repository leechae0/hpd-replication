# HPD experiment code: internal notes (earlier draft)

English translation of the authors' internal notes, originally written in Korean
(`archive/docs_ko/CODE_GUIDE_ko.md` in earlier releases). They describe an earlier state of the code
and are kept for transparency only; the current layout and commands are in the top-level
`README.md`. Numbers below are from that earlier state and are not the numbers of the PeerJ
manuscript.

The observer is a **real LLM**, the MR transformations use **real natural-language processing
(spaCy)**, and the data are read by **parsers for the real files**. Experiments 1 and 2 run without
an LLM (free, offline); Experiment 3 needs an LLM API key.

## Directory layout
```
hpd/
├─ llm.py            real LLM backends (Groq/OpenAI/Anthropic) + Observer (judge/elicit/explain, r-vote)
├─ datasets.py       real data loaders: Patil, HotpotQA, MuSiQue, MAGPIE + Patil→dialogue conversion (real LLM)
├─ mr_nlp.py         real MR transformations: MR4 = NER + coreference + substitution · MR3 = retrieval + unrelatedness filter · MR2 = structural deletion · MR1 = lexical substitution
├─ search.py         search algorithms: backward_search (one minimal set) · enumerate_all (complete enumeration) · jaccard
├─ negatives.py      builds three kinds of negative (non-leaking) scenarios for dialogue-level detection
├─ exp1_search.py    ▶ Experiment 1 — search accuracy, robustness, scalability (controlled oracle)
├─ exp2_certify.py   ▶ Experiment 2 — validity of metamorphic validation (real spaCy MRs)
├─ exp3_endtoend.py  ▶ Experiment 3-1 — real-LLM end-to-end (induction → search → validation + baselines, monotonicity, real-spurious coverage)
└─ exp3_detect.py    ▶ Experiment 3-2 — dialogue-level leakage detection (pool of positives + negatives; precision/recall/F1/false-positive rate)
```

## Role of each module

**llm.py** — `make_backend("groq"|"openai"|"anthropic")` creates a real backend.
`Observer.judge_excerpt` (can z be concluded from the excerpt?), `elicit` (open induction, union
over four framings), `explain` (self-explanation baseline). Every judgment is a majority vote over
r calls. There is no mock backend.

**datasets.py** — unifies four datasets as a common `Scenario` (units, gt, secret). `gt` (the
annotated minimal set) exists only for Patil (`defense`), HotpotQA (`supporting_facts`) and
MuSiQue (`is_supporting`); it is empty for MAGPIE (qualitative demonstration).
`scenario_to_dialogue(backend, s)` turns a Patil scenario into a natural-language dialogue with a
real LLM (one cue = one utterance, labels preserved).

**mr_nlp.py** — the four transformations and a deterministic check for each (`verify_mr1/3/4`).
MR4 finds persons with spaCy NER and speaker prefixes, links pronouns and possessives to them by
rules, and substitutes deterministically. MR3 retrieves sentences that share no content word with
the dialogue from an unrelated corpus and inserts them. MR1 is lexical substitution (a baseline);
a stronger version can swap in a paraphrase model or an LLM.

**search.py** — works on the observer's judgment function `V(S) -> {0,1}`. `backward_search`
removes each cue once from the full set and recovers one minimal supporting set with |U|+1 queries
(= MR2). `enumerate_all` enumerates all minimal sets completely (for small instances).

## Running

### Experiment 1 (free, offline)
```bash
python3 exp1_search.py
# with the real QA files as well:
python3 exp1_search.py --hotpotqa data/hotpot_dev_distractor_v1.json \
                       --musique  data/musique_ans_v1.0_dev.jsonl
```
The observer is a **controlled oracle** that knows the annotation (not an LLM). Noise ε
(0/0.05/0.10/0.20) is injected and judgments are majority votes over r (1/5/7/9); results are
the mean ± standard deviation over 5 seeds, reported with random and exhaustive baselines.

### Experiment 2 (free, offline, needs spaCy)
```bash
pip install spacy && python -m spacy download en_core_web_sm
python3 exp2_certify.py
```
The transformations run with real spaCy. A controlled observer (1 genuine + 4 spurious kinds)
yields [0] implementation check, [1] soundness, [2] coverage, [3] non-redundancy.

### Experiment 3 (real LLM)
```bash
pip install openai spacy && python -m spacy download en_core_web_sm
export GROQ_API_KEY=<key>
python3 exp3_endtoend.py --backend groq --limit 20 --r 3     # 3-1 pilot
python3 exp3_endtoend.py --backend groq --r 5                # 3-1 full run
python3 exp3_endtoend.py --backend openai --model gpt-4o-2024-08-06   # 3-1 frontier model
```
(3-1) reports induction recall (A/B/C gate), search Jaccard, measured soundness,
monotonicity-violation rate, two baselines, and real-spurious coverage.

```bash
# (3-2) dialogue-level detection — pick out leaking dialogues from a pool of positives (Patil) and negatives (non-leaking)
python3 exp3_detect.py --backend groq --limit 20 --r 3
python3 exp3_detect.py --backend groq --limit 20 --r 3 --magpie data/magpie.jsonl  # add natural benign dialogues
```
(3-2) Negatives are built by negatives.py (sensitive combination removed / cues shuffled /
natural benign dialogues). Induction → search → validation is applied to each dialogue; a dialogue
is flagged as leaking if a validated path remains. Reports dialogue-level precision, recall, F1
and false-positive rate, and for true positives the secret match and path Jaccard.

## Offline measurements (code at that time)
- Experiment 1 (Patil): noise-free 116/116; strong noise (ε = 0.20) exact recovery 96.4 ± 3.8,
  Jaccard 0.927; 9 queries vs 256 exhaustive; random 0.431.
- Experiment 2 (real spaCy): [0] MR4 and MR3 116/116, MR1 111/116; [1] soundness 1.000;
  [2] coverage 75.0% (co-occurrence not detected); [3] non-redundancy 116/116 each.
- Experiment 3: needs a real LLM key (induction recall is the gate metric).

## Note: the "controlled oracle" and "controlled observer" are not mocks
The oracle of Experiment 1 and the genuine/spurious observers of Experiment 2 are **experimental
devices in a controlled setting** that know the annotation (they compare against it and inject
genuine or spurious behaviour). This is by design, to verify the method; the transformations
themselves are real NLP code. Real LLM observers are used in Experiment 3.
