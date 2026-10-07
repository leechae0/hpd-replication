# HPD — Hidden-Path Discovery: replication package

Code, prompts, dialogue construction rules, generated dialogue data, per-scenario judgment logs,
and human annotation sheets for the paper

> *Path-Level Auditing of Compositional Privacy Leakage in Multi-Agent Large Language Model
> Dialogues* (submitted to PeerJ Computer Science).

Every number in the paper and its Supplemental Information can be regenerated from this repository.
The controlled checks run offline in a few minutes. The LLM-observer experiments need API access, but
their complete per-scenario logs are included, so every table can be recomputed without re-querying
the models.

## Where each number in the PeerJ manuscript comes from

All commands are run from `code/`. No command below calls a model.

| Manuscript | Command | Output |
|---|---|---|
| Table 2 (induction, paths returned, path length, agreement) and Table 3 (validation) | `python3 peerj_tables.py --json ../results/peerj_tables.json` (reads `results/logs/*_r5_mr4fix.jsonl` and `results/induction/semantic_scores.json`) | `results/peerj_tables.json` |
| Table 2, human-judged induction | `python3 exp3_induction.py score` (reads `results/induction/annotation_sheet.csv`) | `results/induction/semantic_scores.json` |
| Section 4.4, paired tests between observers | `python3 exp3_stats.py --suffix _mr4fix` | `results/exp3_stats_mr4fix.json` |
| Section 4.4 and Section S3, removal order (Table S1) | `python3 exp3_order.py --compare` | `results/exp3_order_summary.json` |
| Section 4.4 and Section S4, repeated judgments (Table S2) | `python3 -c "import exp3_rsweep as R; R.report(R.load_progress())"` (with `rsweep_progress.jsonl` in `code/`) | `results/logs/exp3_rsweep_report.txt` |
| Section 4.5 and Section S5, controlled checks (Table S3) | `python3 exp1_table3.py ...` and `python3 exp2_certify.py` (see below) | `results/logs/exp1_table3_run.txt`, `results/logs/exp2_run.txt` |
| Section S2, transformation audit and MR3 similarity | `results/mr_fidelity/fidelity_sheet.csv`; `python3 exp3_mr3_similarity.py` | `results/logs/exp3_mr3_similarity_run.txt` |
| Section S2, MR4 correction | `python3 exp3_mr4_rerun.py --report-only ...` | `results/logs/*_r5_mr4fix.jsonl` |
| Section S6, multiple paths and blocking (Tables S4, S5) | `python3 exp_c3_multipath.py`; `exp3_block.py` (part a, interactive run, no saved log) | `results/logs/exp_c3_multipath_run.txt`, `results/c3_results.json` |

Second annotator (planned): `python3 make_annotation_xlsx.py --mode sample --suffix _2차판정자` builds the
two annotator workbooks in the same layout as the first pass, restricted to the labelled sample
(108 induction rows, 90 transformation rows) and without the first labels. After the second annotator
returns them, `python3 collect_annotations.py task1 <file> --as ann2 --subset` (and `task2`) writes the
labels into column `ann2` and prints raw agreement and Cohen's kappa against `final`. With
`--mode filled` the same script rebuilds the first annotator's sheets with the recorded labels.

Prepared for the review round, not used in the manuscript: `exp3_table4.py` (message-level versus
path-level auditing). `second_annotator.py` and `annotation2/` are an earlier, simpler version of the
second-annotator sheets and are superseded by the commands above.

The sections below keep the original documentation, which uses the numbering of an earlier draft
(Tables III to VI, Appendices A to C).

## Layout

```
code/        experiment scripts (Python 3.10+)
data/        Patil scenarios (downloaded by code/fetch_patil.py), rendered dialogues, manifest
docs/        prompts (verbatim Korean + English gloss), dialogue construction rules, code guide
results/     logs behind every reported number (results/logs) and pilot/debug runs (results/dev_logs)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # Experiment 2 and the MR transformations
cd code
python3 fetch_patil.py                          # 119 Patil scenario files -> data/patil/ (MD5-verified)
python3 fetch_datasets.py                       # HotpotQA + MuSiQue (Experiment 1 only; needs huggingface_hub, pyarrow)
```

`fetch_datasets.py` writes `data/hotpot_dev_distractor_v1.json` (HotpotQA distractor validation,
7,405 questions) and `data/musique_ans_v1.0_dev.jsonl` (MuSiQue-Answerable validation, 2,417
questions) in the formats the loaders expect. MD5 of the files we used:
`7902702945a42ae3593e5e7c073a505e` (HotpotQA) and `923941264c829ce44bdd6648573f6bf5` (MuSiQue).

## Reproducing the tables

All commands are run from `code/`.

| Paper | Command | Log |
|---|---|---|
| Table III (Exp. 1) | `python3 exp1_table3.py --hotpotqa ../data/hotpot_dev_distractor_v1.json --musique ../data/musique_ans_v1.0_dev.jsonl` | `results/logs/exp1_table3_run.txt` |
| Table IV (Exp. 2) | `python3 exp2_certify.py` | `results/logs/exp2_run.txt` |
| Table V (Exp. 3) | `python3 exp3_compare.py` (reads `e2e_progress_*.jsonl`; copy them from `results/logs/` into `code/`) | `results/logs/exp3_compare_output.txt` |
| Table A.1 | `python3 -c "import exp3_rsweep as R; R.report(R.load_progress())"` (with `rsweep_progress.jsonl` in `code/`) | `results/logs/exp3_rsweep_report.txt` |
| Table C.1, value-exposed row | `python3 exp3_detect.py --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --r 5 --raw` | `results/logs/exp3_detect_value_exposed.txt` |
| Tables C.2, C.3 (Appendix C.3) | `python3 exp_c3_multipath.py` (controlled simulation, about 20 s) | `results/logs/exp_c3_multipath_run.txt`, `results/c3_results.json` |
| Table VI (paired observer comparison) | `python3 exp3_stats.py` (reads `results/logs/e2e_progress_*_r5.jsonl`; exact McNemar with Holm correction, paired bootstrap for Jaccard) | `results/logs/exp3_stats_run.txt`, `results/exp3_stats.json` |
| Appendix C.4 baselines | `python3 exp3_compare.py` (rows "직접질문" = ask-the-model, "leave-one-out") | `results/logs/exp3_compare_output.txt` |

Experiment 1 is deterministic (seeds 0–4) and takes about 10 s. Experiment 2 takes about
2–3 min (spaCy). To re-run Experiment 3 from scratch:

```bash
export OPENAI_API_KEY=...          # never commit keys
export OPENROUTER_API_KEY=...
python3 exp3_endtoend.py --backend openai     --model gpt-4o-mini-2024-07-18   --r 5 --raw
python3 exp3_endtoend.py --backend openrouter --model qwen/qwen3.6-27b          --r 5 --raw
python3 exp3_endtoend.py --backend openrouter --model google/gemini-2.5-flash-lite --r 5 --raw
python3 exp3_compare.py
python3 exp3_rsweep.py --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --raw   # Appendix A
python3 exp3_block.py  --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --r 5 --raw  # Appendix C.2
```

Reviewer-requested follow-ups: `exp3_order.py` (removal-order sensitivity with real observers, Appendix C.5: first 30 scenarios x 12 orders, judgments cached per cue subset; outputs `results/logs/order_progress_<tag>.jsonl`, `results/logs/order_cache_<tag>.json`, and with `--compare` `results/exp3_order_summary.json`), `exp3_induction.py` (secret induction re-scored by a human annotator on a stratified sample; `results/induction/annotation_sheet.csv` holds the 108 labels and a `note` column for A/X rows, `score` writes `results/induction/semantic_scores.json`), `exp3_mr4_rerun.py` (MR4 re-judged on 2026-09-22 with the corrected name filter in `mr_nlp.py`; the paper's Table V MR4/final rows and Table VI come from `results/logs/*_mr4fix.jsonl` and `results/exp3_stats_mr4fix.json`, while the pre-correction logs `*_r5.jsonl` and `results/exp3_stats.json` are kept as run), and `exp3_mr_fidelity.py` (30-scenario human audit sheet of MR1/MR3/MR4 transformation fidelity, `results/mr_fidelity/fidelity_sheet.csv`, filled: 90/90 faithful), and `exp3_mr3_similarity.py` (embedding-based check that MR3 insertions are unrelated to the excerpt; output in `results/logs/exp3_mr3_similarity_run.txt`; needs `en_core_web_md`). PeerJ revision: `exp3_table4.py` (message-level versus path-level auditing with the benchmark secret fixed, 116 positives plus 30 hard and 15 mixed negatives; outputs `results/logs/table4_<tag>.jsonl`, a judgment cache `results/logs/table4_cache_<tag>.jsonl`, and with `--summary` `results/table4_summary.json`) and `second_annotator.py` (blind workbook for a second annotator of the induction and fidelity sheets, and Cohen's kappa; `annotation2/`). See the docstrings.

Runs are resumable: each script appends one JSON line per scenario to a progress file and skips
completed scenarios when restarted. Because the observers are external API models, re-running
will not reproduce the logged judgments bit for bit; the logs in `results/logs/` are the runs the
paper reports.

## Observers and decoding settings (Experiment 3)

| Observer | Access | Model identifier | Settings |
|---|---|---|---|
| GPT-4o-mini | OpenAI API | `gpt-4o-mini-2024-07-18` | temperature 0 |
| Qwen3.6-27B | OpenRouter | `qwen/qwen3.6-27b` | temperature 0, `max_tokens` 800, reasoning disabled |
| Gemini 2.5 Flash-Lite | OpenRouter | `google/gemini-2.5-flash-lite` | temperature 0, `max_tokens` 800, reasoning disabled |

Every binary judgment is asked r = 5 times and decided by majority vote (`llm.Observer`). All
prompts are in Korean and are listed verbatim in `docs/PROMPTS.md`. Dialogues are rendered from the
Patil records deterministically (`--raw`; see `docs/DIALOGUE_RULES.md`); the rendered dialogues used
in the paper are in `data/dialogues/patil_dialogues_116.jsonl`.

## Notes on the logs

* `results/logs/e2e_progress_*_r5.jsonl` — one record per scenario and observer: induction gate
  (`abc`), returned path (`S`), Jaccard to the ground truth (`jac`), query count (`q`), the five
  validation outcomes (`cd`), monotonicity-violation rate (`mono`), and the C.4 baselines
  (`b_ask`, `b_loo`) where they were run (GPT-4o-mini: 104 scenarios; Qwen3.6-27B: the first 30).
* `results/logs/rsweep_progress.jsonl` — Appendix A: validation of the 19 GPT-4o-mini paths
  re-judged with r ∈ {1, 3, 5, 9}.
* `results/logs/exp1_run_2026-08-23.txt` — the original Experiment 1 run; identical to
  `exp1_table3_run.txt` except that MuSiQue used the first 500 questions instead of the
  hop-stratified 1,405 reported in the paper.
* `results/dev_logs/` — pilot, smoke and debugging runs kept for transparency; none of the
  reported numbers come from them. The OpenAI organization id in one error message was redacted.
* Table C.1 (rewritten-input rows) and Appendix C.2 were run interactively without a saved log;
  the scripts (`exp3_detect.py` without `--raw`, `exp3_block.py`) are included.
* Appendix C.3 (`exp_c3_multipath.py`) is a deterministic controlled simulation (seeds 0–4); the
  numbers in the paper are those in `results/logs/exp_c3_multipath_run.txt`.

## Change relative to the runs on the experiment machine

`exp2_certify.py`: the "surface" controlled observer is keyed to the phrase `earlier today`, which
is the phrase MR1 rewrites in the current dialogue rendering (`datasets.py`). The earlier version
was keyed to a phrase that the current rendering no longer contains, so the MR1-ablation row of
Table IV was not exercised (`results/dev_logs/exp2_run_2026-08-23_before_fix.txt` shows the
pre-fix output). Table IV itself is unchanged.

## Data attribution

The Patil scenarios are from V. Patil, E. Stengel-Eskin and M. Bansal, *The Sum Leaks More Than
Its Parts* (arXiv:2509.14284), https://github.com/Vaidehi99/MultiAgentPrivacy. They are not
redistributed here; `fetch_patil.py` downloads them and checks them against
`data/patil_manifest.json`. HotpotQA (Yang et al., 2018) and MuSiQue (Trivedi et al., 2022) are
downloaded from their Hugging Face mirrors by `fetch_datasets.py`.

## License

MIT (see `LICENSE`) for the code and documentation in this repository. Third-party datasets keep
their own terms.
