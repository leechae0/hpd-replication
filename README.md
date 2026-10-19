# HPD (Hidden-Path Discovery): replication package

## Title

**HPD (Hidden-Path Discovery) replication package** for

> Chaeyoung Lee and Sooyong Park. *Path-Level Auditing of Compositional Privacy Leakage in
> Multi-Agent Large Language Model Dialogues.* Submitted to PeerJ Computer Science.

Archived on Zenodo: https://doi.org/10.5281/zenodo.23315182 (concept DOI, resolves to the latest
version; the first release, v1.0-peerj, is https://doi.org/10.5281/zenodo.23315183).
Mirror: https://github.com/leechae0/hpd-replication.

## Description

In a multi-agent system built on large language models (LLMs), a sensitive fact can leak even when
no single message states it: several individually harmless remarks (cues) can together let an
observer infer it. HPD is a black-box auditing method that, given a dialogue log and an observer
model, (1) asks the observer which sensitive conclusions it draws, (2) removes cues one at a time
until none can be dropped, leaving a minimal cue combination (a path), and (3) tests that path for
sufficiency, compositionality, and invariance under paraphrasing, irrelevant-context insertion and
entity substitution.

This package contains the code, observer prompts, rendered dialogues, per-scenario judgment logs of
the three LLM observers, and both annotators' human labels. Every number in the manuscript and its
Supplemental Information can be recomputed from the included logs **without calling any model**
(see Usage Instructions). Re-running the observer experiments themselves needs API keys.

**Language.** Everything in this repository is in English except text that was an experimental
input, the annotation instrument, or raw recorded data: the observer prompts (Korean instructions;
the dialogues and secrets are the English originals of the benchmark), the annotator workbooks and
instructions, a few annotator memos, some observer outputs, and archived development logs.
`docs/CODEBOOK.md` lists every such string with an English gloss, `docs/PROMPTS.md` gives each
prompt verbatim with an English gloss, and `annotation/annotator_guide_en.md` translates the
annotator instructions in full.

## Dataset Information

| Data | Location | Content | Source and licence |
|---|---|---|---|
| Patil et al. benchmark scenarios (119 files, 116 used) | `data/patil/` (downloaded, not redistributed) | per participant: named tables (the cues), the sensitive table combination (`defense`), and the sensitive conclusion (the secret) | V. Patil, E. Stengel-Eskin and M. Bansal, *The Sum Leaks More Than Its Parts* (arXiv:2509.14284), https://github.com/Vaidehi99/MultiAgentPrivacy; fetched and MD5-checked by `code/fetch_patil.py` against `data/patil_manifest.json` |
| Rendered dialogues | `data/dialogues/patil_dialogues_116.jsonl` | the 116 dialogues exactly as shown to the observers; fields `sid`, `secret`, `gt` (annotated cue indices), `turns` (`speaker`, `text`, `unit_idx`) | derived from the Patil scenarios by the deterministic rendering in `docs/DIALOGUE_RULES.md`; regenerate with `code/export_dialogues.py` |
| Observer judgment logs | `results/logs/` | one JSON line per scenario and observer; fields described under Log files below | produced by this study |
| Secret-induction candidates and human labels | `results/induction/` | candidate conclusions per scenario and observer; human labels (`ann1`, `ann2`, `final`, `note`) | produced by this study |
| Transformation-fidelity labels | `results/mr_fidelity/` | 90 transformed excerpts with human labels | produced by this study |
| Annotator workbooks | `annotation/` | the workbooks the two annotators filled in | produced by this study; see `annotation/README.md` |
| HotpotQA (distractor, dev) and MuSiQue-Answerable (dev) | `data/` (downloaded, not redistributed) | used only for the controlled checks of Section S5 | Yang et al. (2018); Trivedi et al. (2022); fetched from Hugging Face by `code/fetch_datasets.py` |

The study involved no personal data: all dialogues are synthetic benchmark scenarios.

## Code Information

All scripts are in `code/` (Python 3.10 or later) and run from that folder.

| Script | Role |
|---|---|
| `llm.py` | model backends (OpenAI, OpenRouter, and others) and the observer: binary judgment with r-fold majority vote, secret induction; the observer prompts |
| `datasets.py`, `fetch_patil.py`, `fetch_datasets.py`, `export_dialogues.py` | data loading, download with checksums, dialogue rendering |
| `search.py` | path discovery (one-at-a-time cue removal) and complete enumeration for small cases |
| `mr_nlp.py` | the transformations MR1 (rephrasing), MR3 (irrelevant-sentence insertion), MR4 (entity substitution) with spaCy, and their deterministic checks |
| `exp3_endtoend.py`, `exp3_mr4_rerun.py` | main observer experiment: induction, discovery, validation, monotonicity probe (Sections 4.1 to 4.4) |
| `exp3_order.py`, `exp3_rsweep.py`, `exp3_block.py`, `exp_c3_multipath.py` | removal-order sensitivity, repeated judgments, blocking, multiple paths (Sections S3, S4, S6) |
| `exp3_induction.py`, `make_annotation_xlsx.py`, `collect_annotations.py` | human audit: candidate dump, annotator workbooks, label collection and agreement |
| `exp1_search.py`, `exp1_table3.py`, `exp2_certify.py` | controlled checks with a rule-based judge (Section 4.5, Section S5) |
| `exp3_stats.py`, `exp3_mr3_similarity.py`, `exp3_mr_fidelity.py` | statistics and audits of the transformations |
| `peerj_tables.py`, `peerj_sensitivity.py` | Tables 2 and 3, Section S7 and Table S6 from the logs |

### Repository layout

```
code/         all scripts used in the paper (Python 3.10 or later)
data/         rendered dialogues and the manifest of the Patil benchmark files (data/README.md)
docs/         PROMPTS.md: observer prompts, verbatim Korean with English gloss
              DIALOGUE_RULES.md: how dialogues, cues, secrets and transformations are built
              CODEBOOK.md: English glosses for every Korean string left in the repository
results/      logs and summaries behind every reported number
  logs/         per-scenario judgment logs of the observer runs
  induction/    secret-induction candidates and human labels
  mr_fidelity/  human audit of the transformations
annotation/   annotator workbooks, annotator guide (Korean original and English translation),
              second annotator's returned workbooks
archive/      analyses of an earlier draft and development logs; not used in the paper
```

## Usage Instructions

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm       # needed by code/mr_nlp.py, which most scripts import
cd code
python3 fetch_patil.py                         # 119 Patil scenario files -> data/patil/ (MD5-checked)
```

`fetch_datasets.py` (HotpotQA and MuSiQue) is needed only to re-run the controlled checks of
Section S5. `exp3_mr3_similarity.py` additionally needs `en_core_web_md`.

### Quick check (no model calls, under a minute)

```bash
cd code
python3 peerj_tables.py         # Tables 2 and 3, Section 4.4 monotonicity, Section S4 four-of-five rates
python3 peerj_sensitivity.py    # Section S7 and Table S6 (12 scenarios with unsupported secrets excluded)
```

### Where each number comes from

All commands run from `code/`. None of them calls a model.

| Manuscript | Command | Reads | Writes |
|---|---|---|---|
| Table 2, Table 3; Section 4.4 monotonicity; Section S4 four-of-five rates | `python3 peerj_tables.py --json ../results/peerj_tables.json` | `results/logs/*_r5_mr4fix.jsonl`, `results/induction/semantic_scores.json` | `results/peerj_tables.json` |
| Table 2, human-judged induction; Section 4.2; Section S2 (a) | `python3 exp3_induction.py score` | `results/induction/annotation_sheet.csv` | `results/induction/semantic_scores.json` |
| Sections 4.2 and 4.3, inter-annotator agreement | `python3 collect_annotations.py task1 ../annotation/second_annotator/task1_induction_second_annotator.xlsx --as ann2 --subset` (and `task2` with `task2_transformation_second_annotator.xlsx`) | the workbook and the sheet CSV | column `ann2`, `*_ann2_disagreements.csv` |
| Section 4.3, transformation fidelity; Section S2 (b) | human labels | `results/mr_fidelity/fidelity_sheet.csv` | |
| Section S2 (b), MR3 similarity | `python3 exp3_mr3_similarity.py` | `results/logs/*_r5_mr4fix.jsonl` | `results/logs/exp3_mr3_similarity_run.txt` |
| Section 4.4, paired tests between observers | `python3 exp3_stats.py --suffix _mr4fix` | `results/logs/*_r5_mr4fix.jsonl` | `results/exp3_stats_mr4fix.json` |
| Section 4.4 and Section S3, removal order (Table S1) | `python3 exp3_order.py --compare` | `results/logs/order_progress_*.jsonl`, `order_cache_*.json` | `results/exp3_order_summary.json` |
| Section 4.4 and Section S4, repeated judgments (Table S2) | copy `results/logs/rsweep_progress.jsonl` into `code/`, then `python3 -c "import exp3_rsweep as R; R.report(R.load_progress())"` | `rsweep_progress.jsonl` | `results/logs/exp3_rsweep_report.txt` |
| Section 4.4 and Section S7 (Table S6) | `python3 peerj_sensitivity.py --json ../results/peerj_sensitivity.json` | logs and `annotation_sheet.csv` (`note` column) | `results/peerj_sensitivity.json` |
| Section 4.5 and Section S5, controlled checks (Table S3) | `python3 exp1_table3.py --hotpotqa ../data/hotpot_dev_distractor_v1.json --musique ../data/musique_ans_v1.0_dev.jsonl` and `python3 exp2_certify.py` | data files | `results/logs/exp1_table3_run.txt`, `results/logs/exp2_run.txt` |
| Section 5.3 and Section S6 (b), multiple paths and blocking (Tables S4, S5) | `python3 exp_c3_multipath.py` | Patil files | `results/c3_results.json`, `results/logs/exp_c3_multipath_run.txt` |
| Section S6 (a), blocking with GPT-4o-mini | `exp3_block.py` (calls the model; run interactively, no saved log) | | |
| Proposition 1, Section S1 | proofs; no code | | |

We re-ran every command above, except the HotpotQA and MuSiQue checks, from a fresh clone and
obtained the committed outputs. The only differences were the key order in
`semantic_scores.json` and the Wilcoxon p-values in `exp3_stats_mr4fix.json`; these are filled in
only when `scipy` is installed and are not reported in the paper.

### Re-running the observer experiments (calls the models)

Set keys in the shell only; never write them into a file or commit them.

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
cd code
# main run: induction, discovery, validation, monotonicity probe (Sections 4.2 to 4.4)
python3 exp3_endtoend.py --backend openai     --model gpt-4o-mini-2024-07-18      --r 5 --raw
python3 exp3_endtoend.py --backend openrouter --model qwen/qwen3.6-27b             --r 5 --raw
python3 exp3_endtoend.py --backend openrouter --model google/gemini-2.5-flash-lite --r 5 --raw
# MR4 re-judged with the corrected name filter -> results/logs/*_r5_mr4fix.jsonl (Section S2)
python3 exp3_mr4_rerun.py --backend openai --model gpt-4o-mini-2024-07-18 --r 5 --raw      # and the other two observers
# removal-order sensitivity (Section S3): first 30 scenarios, 10 random orders + dialogue order + reverse
python3 exp3_order.py --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --orders 10 --r 5 --raw
# repeated judgments (Section S4) and blocking (Section S6 a), GPT-4o-mini only
python3 exp3_rsweep.py --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --raw
python3 exp3_block.py  --backend openai --model gpt-4o-mini-2024-07-18 --limit 30 --r 5 --raw
# secret-induction candidates for the human audit (Section S2 a)
python3 exp3_induction.py dump --backend openai --model gpt-4o-mini-2024-07-18 --raw      # and the other two observers
python3 exp3_induction.py sheet
```

Runs are resumable: each script appends one JSON line per scenario and skips completed scenarios
on restart. Hosted models change and are not deterministic even at temperature 0, so a re-run
will not reproduce the logged judgments exactly; the logs in `results/logs/` are the runs the
paper reports.

## Requirements

* Python 3.10 or later.
* Python packages in `requirements.txt`: `openai>=1.0`, `spacy>=3.7,<3.8`, `numpy`, `openpyxl`;
  spaCy models `en_core_web_sm` (most scripts) and `en_core_web_md` (`exp3_mr3_similarity.py` only).
* Optional: `scipy` (Wilcoxon p-values in `exp3_stats.py`, not reported), `huggingface_hub` and
  `pyarrow` (`fetch_datasets.py`), `anthropic` (unused backend).
* Re-running the observer experiments needs an OpenAI API key and an OpenRouter API key. The
  quick check and every command in "Where each number comes from" need no key and no network
  access beyond the initial data download.

## Methodology

1. **Data processing.** Each Patil scenario is turned into a dialogue with one utterance per table
   (cue); the annotated cue set is the sensitive combination named in `defense`, and the secret is
   the scenario's sensitive conclusion. Scenarios without a proper sensitive combination are
   dropped (116 of 119 remain). The rendering is deterministic and value-preserving
   (`docs/DIALOGUE_RULES.md`).
2. **Induction.** The observer lists the conclusions it can draw from the full dialogue (three
   framings, unioned). A scenario passes the gate if a candidate matches the benchmark secret
   lexically (A) or the observer affirms the secret on the full dialogue (B).
3. **Discovery.** Starting from all cues, each cue is removed once and kept out if the observer
   still concludes the secret; the result is a minimal path.
4. **Validation.** The path is checked for sufficiency, compositionality (no single cue suffices),
   and invariance under MR1, MR3 and MR4. Every binary judgment is the majority of r = 5 calls at
   temperature 0.
5. **Human audit.** Two annotators labelled secret induction (108 sampled rows of 348) and
   transformation fidelity (90 rows); agreement is reported as raw agreement and Cohen's kappa.
6. **Analysis.** Rates, confidence intervals and the sensitivity analyses are computed from the
   logs by the scripts listed above.

### Observers and settings

| Observer | Access | Model identifier | Settings |
|---|---|---|---|
| GPT-4o-mini | OpenAI API | `gpt-4o-mini-2024-07-18` | temperature 0 |
| Qwen3.6-27B | OpenRouter | `qwen/qwen3.6-27b` | temperature 0, `max_tokens` 800, reasoning disabled |
| Gemini 2.5 Flash-Lite | OpenRouter | `google/gemini-2.5-flash-lite` | temperature 0, `max_tokens` 800, reasoning disabled |

Runs took place between 20 August and 22 September 2026. Every binary judgment is asked r = 5
times and decided by majority vote (`llm.Observer`). OpenRouter requests were not pinned to one
upstream provider (`provider: {allow_fallbacks: true}` in `code/llm.py`), so the serving provider
may have varied between calls.

### Log files

`results/logs/e2e_progress_<observer>_r5_mr4fix.jsonl` holds one record per scenario:

| Field | Meaning |
|---|---|
| `sid` | Patil scenario id |
| `abc` | induction gate: A = an induced candidate matches the secret lexically; B = no match, but the observer affirms the secret on the full dialogue; C = neither (no path is searched) |
| `S` | returned path (cue indices), or null |
| `jac` | Jaccard similarity between `S` and the annotated cue set |
| `q` | judgments used by discovery |
| `cd` | the five validation checks: `base` (sufficiency), `single` (single-cue check), `MR1`, `MR3`, `MR4` |
| `cert`, `cert4` | all five checks passed; at least four passed |
| `mono` | monotonicity-violation rate of the probe |
| `MR4_old`, `cert_old`, `mr4_old_nonname`, `mr4_names` | MR4 before the name-filter correction, and the corrected renaming map |
| `b_ask`, `b_loo` | baselines of an earlier draft; not reported in the paper |

`e2e_progress_<observer>_r5.jsonl` are the same runs before the MR4 correction; they are the input
of `exp3_mr4_rerun.py`. A path counts as returned only when `abc` is A or B and `S` is not null.

### Human annotation

See `annotation/README.md`. In short, `results/induction/annotation_sheet.csv` (108 labelled rows
out of 348) and `results/mr_fidelity/fidelity_sheet.csv` (90 rows) hold the first annotator's
labels (`ann1`, `final`), the second annotator's labels (`ann2`) and notes. The first annotator
started from preliminary labels drafted by an LLM assistant and set every final label; the second
annotator worked blind and without AI tools.

## Citations

If you use this package, please cite the article:

> Chaeyoung Lee and Sooyong Park. Path-Level Auditing of Compositional Privacy Leakage in
> Multi-Agent Large Language Model Dialogues. PeerJ Computer Science (submitted), 2026.
> Replication package: https://doi.org/10.5281/zenodo.23315182

Data and tools used:

* V. Patil, E. Stengel-Eskin and M. Bansal. *The Sum Leaks More Than Its Parts: Compositional
  Privacy Risks and Mitigations in Multi-Agent Collaboration.* arXiv:2509.14284, 2025.
* Z. Yang et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.*
  EMNLP 2018.
* H. Trivedi et al. *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL 2022.
* M. Honnibal et al. *spaCy: Industrial-strength Natural Language Processing in Python.*
  https://spacy.io

## License & Contribution Guidelines

The code and documentation are released under the MIT License (`LICENSE`). The third-party
datasets are not redistributed and remain under their own licences. This repository is a frozen
replication package for the article; please report problems or questions through the GitHub issue
tracker (https://github.com/leechae0/hpd-replication/issues) or to the corresponding author.
