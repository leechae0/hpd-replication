# English codebook for the Korean text in this repository

All code comments, docstrings, console output, documentation and README files are in English. The
only Korean text left in the repository is listed below. Each kind is either an experimental input,
the instrument the annotators used, or raw data recorded during the study; translating it in place
would change what was actually used or recorded, so it is kept verbatim and glossed here.

| Kind | Where | Why it stays in Korean | English |
|---|---|---|---|
| Observer prompts | `code/llm.py`, `code/datasets.py` | sent to the observer models verbatim | section 1; `docs/PROMPTS.md` |
| Two other strings sent to observers | `code/datasets.py`, `code/exp3_endtoend.py` | sent to the observer models verbatim | section 1 |
| Annotation workbooks | `annotation/workbooks/`, `annotation/second_annotator/`, `code/make_annotation_xlsx.py` | the instrument the Korean-speaking annotators worked in | section 2; `annotation/annotator_guide_en.md` |
| Annotator instructions | `annotation/annotator_guide_ko.md` | the instructions as given | `annotation/annotator_guide_en.md` (full translation) |
| Annotator memos | second annotator's Task 1 workbook; `memo_ann2` column of `results/induction/annotation_sheet_ann2_disagreements.csv` | free-text notes written by an annotator | section 3 |
| Observer outputs | `candidates` in `results/induction/*.jsonl` and `results/induction/annotation_sheet.csv` | raw model output | section 4 |
| Regular expressions | `code/exp3_endtoend.py`, `code/make_annotation_xlsx.py` | they match Korean text in observer outputs | section 4 |
| Archived development logs | `archive/dev_logs/`, `archive/results/logs/` | console output of earlier code versions, recorded as is; not used for any reported number | section 5 |

## 1. Strings sent to the observers

The five observer prompts are reproduced with an English gloss in `docs/PROMPTS.md`; the same
glosses appear as comments next to each prompt in `code/llm.py` and `code/datasets.py`. Two further
Korean strings reach the observer:

| String | Where | English | Used for |
|---|---|---|---|
| `질문 '<question>'의 답은 <answer>이다.` | `code/datasets.py`, `load_hotpotqa` | "The answer to the question '<question>' is <answer>." | judged statement z in the HotpotQA controlled check (Section S5) |
| `'<question>?'의 답은 <answer>이다.` | `code/datasets.py`, `load_musique` | "The answer to '<question>?' is <answer>." | judged statement z in the MuSiQue controlled check (Section S5) |
| `(내용 은닉)` | `code/exp3_endtoend.py`, `harvest_coverage` | "(content hidden)" | placeholder text of a content-masked dialogue in the real-spurious coverage check (`harv`, `caught` fields of the logs; not reported in the manuscript) |

The controlled checks of Section S5 use a rule-based judge, not an LLM, so the two judged statements
are never read by a model there; they only matter if `exp3_endtoend.py` is run on those datasets.

## 2. Annotation workbooks

Built by `code/make_annotation_xlsx.py`. Each workbook has three sheets.

| Korean | English |
|---|---|
| `작업1_유도판정.xlsx` | Task 1 secret-induction judgment (file) |
| `작업2_변환판정.xlsx` | Task 2 transformation-fidelity judgment (file) |
| sheet `안내` | Instructions (full translation: `annotation/annotator_guide_en.md`) |
| sheet `작업1_판정` / `작업2_판정` | Task 1 labels / Task 2 labels (the sheet the annotator fills in) |
| sheet `_key` | hidden key mapping row number `No` to (`sid`, `observer`) or (`sid`, `mr`) |

**Task 1 columns** (sheet `작업1_판정`)

| Column | Korean header | English |
|---|---|---|
| A | `No` | row number |
| B | `정답 (이 대화에서 새어 나갈 수 있는 민감한 사실)` | Reference answer (the sensitive fact that could leak from this dialogue) = benchmark secret |
| C | `모델이 낸 결론 문장 (모델이 쓴 순서 그대로)` | Conclusion sentences produced by the model (in the order the model wrote them) |
| D | `결론 문장 (계속) — 길 때만` | Conclusion sentences (continued) — only when long |
| E | `판정 M / X / A` | Label: M (match) / X (mismatch) / A (ambiguous) |
| F | `메모 (선택)` | Memo (optional) |
| G | `참고: 표를 그대로 옮긴 줄 · 짧은 조각` | For reference: lines copied verbatim from a table · short fragments |

Cell prefixes in column G: `[표를 옮긴 줄]` = "[lines copied from a table]", `[짧은 조각 — 보통 무시]` =
"[short fragments — usually ignore]". Column C may read `(문장으로 된 결론 없음 → 오른쪽 '참고' 열을 보고 판단)`
= "(no conclusion in sentence form → decide from the 'reference' column on the right)".

**Task 2 columns** (sheet `작업2_판정`)

| Column | Korean header | English |
|---|---|---|
| A | `No` | row number |
| B | `변환 종류` | Transformation type: `MR1 표현 바꾸기` = MR1 rephrasing; `MR3 무관한 문장 끼워 넣기` = MR3 irrelevant-sentence insertion; `MR4 사람 이름 바꾸기` = MR4 person-name substitution |
| C | `확인할 질문` | Question to check (the three questions are translated in `annotation/README.md`, Task 2 table) |
| D | `변환 전 (원문 전체)` | Before the transformation (full original text) |
| E | `변환 후 (전체) — 빨간 글씨 = 바뀐 곳` | After the transformation (full text) — red text = what changed |
| F | `판정 Y / N` | Label: Y (yes) / N (no) |
| G | `메모 (선택)` | Memo (optional) |
| H | `정답 문장 / (MR4) 이름 바꾼 표` | Reference sentence / (MR4) renaming table |

Cell prefixes in column H: `[원래 정답]` = "[original reference answer]", `[이름 바꾼 뒤 정답]` =
"[reference answer after renaming]", `[이름 바꾼 표]` = "[renaming table]".

**Other workbook text.** `진행 상황` = "Progress"; `입력한 행: n / N (남은 행: k)` = "Rows filled: n / N
(rows left: k)"; drop-down error `입력 오류` / `M/X/A 중 하나만 입력하세요.` = "Input error" / "Enter only
one of M/X/A." (Y/N for Task 2). Input-cell notes: `M = 같은 사실 있음, X = 전부 다른 얘기, A = 관련은
있으나 부족` = "M = the same fact is present, X = everything is about something else, A = related but
insufficient"; `Y = 질문에 '그렇다', N = '아니다'` = "Y = 'yes' to the question, N = 'no'".

The labels themselves (M/X/A, Y/N) are Latin letters, and all labels are consolidated in English
CSV columns (`ann1`, `ann2`, `final`, `note`) under `results/` (see `annotation/README.md`).

## 3. Annotator memos

The second annotator wrote seven optional memos in Task 1 (workbook
`annotation/second_annotator/task1_induction_second_annotator.xlsx`, column F). The first annotator's
notes are already in English (`note` column of `results/induction/annotation_sheet.csv`). The second
annotator wrote no memos in Task 2.

| Row `No` | Memo (Korean, verbatim) | English |
|---|---|---|
| 41 | 프로젝트 내용이 다른거 같음 | The project details seem to be different. |
| 49 | 프로젝트 내용이 다른거 같음 | The project details seem to be different. |
| 61 | were never ordered 이거 없는데 | "were never ordered" — this is not there. |
| 93 | 직원의 집이 어디인지는 나오지 않음 | Where the employee lives is not stated. |
| 105 | hypertension 만 맞음 | Only "hypertension" is correct. |
| 116 | Dallas 아님 Houston 임. | Not Dallas; it is Houston. (also in `annotation_sheet_ann2_disagreements.csv`) |
| 295 | 정보 자체가 연결이 안됨 | The pieces of information do not connect at all. |

## 4. Observer outputs

The observer instructions were in Korean and the dialogues in English, so the observers sometimes
answered the secret-induction prompt (prompt 3) partly in Korean. These candidate conclusions are
raw model output and are kept exactly as returned.

| Observer | Scenarios | Scenarios with at least one Korean candidate | Candidates | Korean candidates |
|---|---|---|---|---|
| GPT-4o-mini | 116 | 23 | 2,671 | 272 (10%) |
| Qwen3.6-27B | 116 | 103 | 4,546 | 1,571 (35%) |
| Gemini 2.5 Flash-Lite | 116 | 101 | 8,911 | 2,841 (32%) |

How this affects the analysis: the lexical induction criterion (`exp3_endtoend.mentions`) counts
shared content words, so a Korean candidate rarely matches an English secret lexically. The human
induction labels (Task 1) do not have this limitation: the annotators are Korean speakers who read
English, and the instructions state that a candidate in Korean counts as a match when it states
the same fact. The regular expressions that include Hangul syllables (`가-힣`) exist so that Korean
words in candidates are tokenized; `범주` = "category" and `결론` = "conclusion" in
`code/make_annotation_xlsx.py` strip a "category ... conclusion ..." prefix some observers used.

## 5. Archived development logs

`archive/dev_logs/` and `archive/results/logs/` hold console output of earlier code versions. They
are not used for any number in the manuscript (see `archive/README.md`) and are kept as recorded. The
console labels they contain are the same as in the current code, which now prints them in English:

| Korean label | English |
|---|---|
| `실험 N 결과 (집계 대상 n개)` | Experiment N results (n scenarios aggregated) |
| `실험 3 종단 — n개 시나리오` | Experiment 3 end-to-end — n scenarios |
| `--- 시나리오 k (sid) ---` | --- scenario k (sid) --- |
| `정답 비밀:` | benchmark secret: |
| `직접 판정(full→z): YES/NO` | direct judgment (full dialogue → z): YES/NO |
| `유도 후보(n개):` | induction candidates (n): |
| `인증 검사:` | validation checks: |
| `[게이트] 유도 A/B/C = … → 재현율(A율) = …` | [gate] induction A/B/C = … → recall (A rate) = … |
| `[탐색] 경로 자카드 평균 … · 질의 평균 …` | [search] mean path Jaccard … · mean queries … |
| `[가정] 단조성 위반율 평균 …` | [assumption] mean monotonicity-violation rate … |
| `[가정] 단조성 표본 없음 — 유도 게이트(A/B)를 통과한 시나리오가 없음` | [assumption] no monotonicity samples — no scenario passed the induction gate (A/B) |
| `[인증] 실측 soundness (전체 통과) k/n` | [validation] measured soundness (all checks passed) k/n |
| `[관계별 불변율]` / `불변율` | [per-check pass rate] / pass rate |
| `[베이스라인] 직접질문 … · leave-one-out … (제안 …)` | [baseline] ask-the-model … · leave-one-out … (proposed …) |
| `[중단] k번째에서 한도/오류로 멈춤:` | [stopped] limit/error at item k: |
| `→ 여기까지 n개 완료분으로 결과를 집계합니다.` | → aggregating the results of the n completed items. |
| `잡음(ε,r) \| 정확 복원율(평균±표준편차) \| 자카드(평균) \| 질의수` | noise (ε, r) \| exact recovery (mean ± sd) \| Jaccard (mean) \| queries |
| `무잡음 / 약 / 중 / 강` | noise-free / weak / medium / strong |
| `[베이스라인] 무작위(크기 제공) 자카드 · 전수 질의수 (제안)` | [baseline] random (size given) Jaccard · exhaustive queries (proposed) |
| `실험 2 (실 spaCy MR)`, `[측정 0–3]` | Experiment 2 (real spaCy MRs), [Measure 0–3] |
| `구현 검증 (변형이 명세대로 — 결정적)` | implementation check (transformations as specified — deterministic) |
| `MR1_인물보존`, `MR4_원래이름_0`, `MR3_추가_무관` | MR1 persons kept, MR4 no original name left, MR3 added sentences unrelated |
| `진성 유지`, `스퓨리어스 검출`, `비중복성` | genuine paths kept, spurious paths caught, non-redundancy |
| `잔여 오류(우연 상관, 정직 보고)` | residual error (coincidental co-occurrence, reported as is) |
| `제거 → … 유형 통과` | removed → … type passes |
| `실험 3-2 대화 수준 탐지 결과`, `탐지 풀: 양성 · 음성` | Experiment 3-2 dialogue-level detection results, detection pool: positives · negatives |
| `혼동행렬`, `정밀도 · 재현율 · F1 · 오탐률` | confusion matrix, precision · recall · F1 · false-positive rate |
| `양성 적중 시 기밀 일치도 · 경로 자카드` | for true positives: secret match · path Jaccard |
| `음성 유형별 오탐` | false positives by negative type |

## 6. Logs of the reported runs

Three logs under `results/logs/` were first recorded with the Korean console messages of the time
and were replaced, for this release, by English output:

| Log | How the English version was produced | Check |
|---|---|---|
| `exp2_run.txt` | re-run of `exp2_certify.py` (deterministic) | the earlier code reproduced the recorded log line for line; the English run gives the same numbers |
| `exp3_rsweep_report.txt` | re-run of `exp3_rsweep.report` on `rsweep_progress.jsonl` (no model calls) | as above |
| `exp1_table3_run.txt` | Patil block: re-run of `exp1_table3.py`; HotpotQA and MuSiQue blocks: the recorded lines with only the labels translated (the data files were not available for the re-run) | every number in the file is unchanged; the Patil block equals the fresh run |

The Korean originals remain in the git history before release v1.1-peerj. All other logs contain no
Korean text.
