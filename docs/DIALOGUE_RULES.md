# Dialogue construction rules and input transformations

Source files: `code/datasets.py`, `code/negatives.py`, `code/mr_nlp.py`, `code/exp3_endtoend.py`.

## 1. From a Patil scenario to a dialogue

Each Patil file (`experiment_<id>_def_adv_both.json`) contains, per participant (`data_distribution`),
a set of named tables with rows, a `defense` string naming the sensitive table combination, and the
sensitive conclusion (`run_2_sensitive.final_inference_result`).

* **Cue = table.** Every table held by a participant becomes one cue (`{"agent", "key", "text"}`),
  so one cue is one utterance. All 116 scenarios used have exactly 8 cues.
* **Ground-truth cue set G** = the tables named in `defense` ("combination of A, B and C is
  sensitive"). Scenarios with no such combination, or whose combination covers every table, are
  dropped; this leaves 116 of the 119 files.
* **Secret z** = `final_inference_result`, truncated before any "suggesting …" clause, so that z
  states the factual combination the cues determine rather than the data-external speculation
  that follows it.
* **Utterance text (value-preserving rendering, used in Experiment 3, Appendices A, C.2, C.4 and
  the "value-exposed" row of Table C.1):**

  ```
  <agent> was reviewing the '<table>' records earlier today. Entries: (<k>=<v>, …); (<k>=<v>, …); ….
  ```

  At most 6 rows of the table are rendered. The dialogue is the list of these utterances in file
  order, one per cue, prefixed with the speaker name when shown to the observer
  (`"<speaker>: <text>"`). This rendering is deterministic; `code/export_dialogues.py` writes all
  116 dialogues to `data/dialogues/patil_dialogues_116.jsonl`.
* **LLM-rewritten rendering (Table C.1 "rewritten input" only):** each utterance is rewritten by
  the observer model itself with the prompt in `PROMPTS.md` §5 (one record per call, no other
  record visible).

HotpotQA and MuSiQue (Experiment 1 only) use paragraphs as cues: HotpotQA's 10 context paragraphs
with the 2 gold paragraphs as G; MuSiQue's ~20 paragraphs with the `is_supporting` paragraphs as G.
Only the combinatorial structure is used there (the oracle never reads text).

## 2. Negative dialogues for per-dialogue screening (Table C.1)

Built by `negatives.py` from the first 30 Patil scenarios (`--limit 30`):

* **Hard negatives (30):** the ground-truth cues are removed and only the remaining distractor
  cues are kept, so no sensitive combination is present but the dialogue looks like a leakage
  dialogue.
* **Shuffled negatives (15):** 3–5 cues sampled from the pool of all cues of different scenarios
  (`random.Random(0)`), so no coherent secret exists.

Negatives carry no secret and no label; the screening pipeline (induction → discovery →
validation on up to 5 induced candidates) is run blind on the mixed pool of 30 positives and 45
negatives.

## 3. Discovery (MR2, backward elimination)

`search.backward_search`: query V_z(U) once; then for each cue u in fixed file order, query
V_z(S∖{u}) and drop u when the judgment stays 1. Exactly |U|+1 queries. Each query is a
majority vote over r = 5 calls of prompt §2 in `PROMPTS.md`.

## 4. Validation checks (`exp3_endtoend.certify`)

Applied to the returned path S*:

1. **Sufficiency re-check:** V_z(S*) = 1 when only the path utterances are shown.
2. **Single-cue check:** V_z({u}) = 0 for every u ∈ S*; paths of size < 2 fail.
3. **MR1 rephrasing** (`mr_nlp.mr1_reframe`): spaCy POS-conditioned lexical substitution from a
   fixed synonym table (e.g. increased→risen, often→frequently, earlier→before, today→this
   morning). Table names and record values are never touched. In the value-preserving rendering
   the only substitutable words are "earlier today" → "before this morning"; this is the
   restricted transformation discussed in Section VI-A. Verified automatically: person names and
   numbers preserved.
4. **MR3 irrelevant-sentence insertion** (`mr_nlp.mr3_inject_noise`): two sentences are chosen
   from an 8-sentence unrelated corpus (marathon route, volcanic soil, telescope, sourdough, subway,
   cranes, mural, solar panels) such that they share no content lemma with the dialogue, and are
   appended as utterances of a new speaker "Owen". Verified automatically: zero content-lemma
   overlap.
5. **MR4 entity substitution** (`mr_nlp.mr4_rename`): PERSON entities (spaCy NER) plus speaker
   prefixes are mapped consistently to new names from {Zara, Kenji, Priya, Owen, Mara, Tobias,
   Lena, Rahul}; the same mapping is applied to the judged statement z. Verified automatically:
   no original name remains; non-name tokens unchanged.

A path passes when all five hold (5/5 criterion). `cert4` records the relaxed 4/5 criterion used in
Appendix A.

## 5. Other measurements in the Experiment 3 logs

* **Monotonicity violation** (`mono`): 8 random pairs (S, S∪{i}) per scenario; a violation is
  V_z(S)=1 and V_z(S∪{i})=0.
* **Baselines** (`b_ask`, `b_loo`, Appendix C.4): ask-the-model line selection (prompt §4) and
  leave-one-out necessity (remove one utterance at a time, keep those whose removal flips the
  judgment), each scored by Jaccard against G.
* **Blocking** (`exp3_block.py`, Appendix C.2): validated path → one exclusion re-search for a
  second path → greedy hitting set B → utterances in B replaced by "(content withheld)" →
  direct judgment and open-ended induction repeated on the masked dialogue.
