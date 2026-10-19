# annotation/

Human audits reported in Sections 4.2, 4.3 and 5.4 and in Section S2 of the Supplemental
Information. The labels themselves are consolidated in two CSV files under `results/`; this folder
keeps the workbooks the annotators worked in.

## Tasks

**Task 1, secret induction (M / X / A).** Each row shows the benchmark secret of one scenario and
the candidate conclusions one observer listed for that scenario. The annotator decides whether
any candidate states the same fact as the secret.

| Label | Meaning |
|---|---|
| M (match) | at least one candidate states the same fact as the secret; wording, language (some candidates are in Korean), length, or referring to a person by ID instead of name do not matter |
| X (mismatch) | every candidate is about something else |
| A (ambiguous) | related, but the candidate does not identify whose fact or what fact, or states only part of it |

The observer is hidden, and rows of the same scenario appear together.

**Task 2, transformation fidelity (Y / N).** Each row shows a dialogue excerpt before and after one
automatic transformation, with the changed text highlighted. The annotator answers the question for
that transformation:

| Transformation | Question (Y if yes) |
|---|---|
| MR1, rephrasing | Did only the wording change, with facts, persons and numbers unchanged? |
| MR3, irrelevant insertion | Is the inserted sentence unrelated to the dialogue and to the secret? |
| MR4, entity substitution | Were only person names changed, consistently, with everything else unchanged? |

Rules given to both annotators: change only the label and memo cells; fill every row; do not
consult each other, search the web, or use AI tools.

## Files

| Path | Content |
|---|---|
| `annotator_guide_ko.md` | the instructions given to annotators, in Korean as they received them |
| `annotator_guide_en.md` | full English translation of `annotator_guide_ko.md` (not shown to the annotators) |
| `workbooks/` | blank workbooks for all rows (348 for Task 1, 90 for Task 2), written by `code/make_annotation_xlsx.py`. File names and sheet text are in Korean, as the annotators saw them: `작업1_유도판정` = Task 1 secret induction, `작업2_변환판정` = Task 2 transformation fidelity; every sheet name and column header is glossed in `docs/CODEBOOK.md` |
| `second_annotator/task1_induction_second_annotator.xlsx` | workbook returned by the second annotator: the 108 sampled Task 1 rows, labelled blind |
| `second_annotator/task2_transformation_second_annotator.xlsx` | workbook returned by the second annotator: the 90 Task 2 rows, labelled blind |

## Where the labels are

| File | Columns |
|---|---|
| `results/induction/annotation_sheet.csv` | 348 rows; `ann1` and `final` (first annotator), `ann2` (second annotator) on the 108 sampled rows; `note` on the 26 rows not labelled M (`benchmark-label-inconsistent` = the benchmark secret is contradicted or unsupported by the scenario records; `model-missed` = the observer missed a supported secret) |
| `results/mr_fidelity/fidelity_sheet.csv` | 90 rows; `ann1`, `final`, `ann2` |
| `results/*/..._ann2_disagreements.csv` | rows where `ann2` differs from `final` |

The first annotator started from preliminary labels drafted by an LLM assistant (Claude,
Anthropic), checked every item and set every final label. The second annotator did not see these
labels or drafts and used no AI tools. `code/collect_annotations.py` reads a returned workbook,
writes its labels into the CSV, and prints raw agreement and Cohen's kappa against `final`.
