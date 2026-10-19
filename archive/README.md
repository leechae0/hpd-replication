# archive/

Material from earlier drafts of the paper and from development. **Nothing here is used for any
number in the PeerJ Computer Science manuscript or its Supplemental Information.** It is kept for
transparency. Scripts in `archive/code/` import modules from `code/` and are not maintained; to
run one, copy it into `code/`.

| Path | What it is |
|---|---|
| `code/exp3_compare.py` | summary table of an earlier draft; replaced by `code/peerj_tables.py` |
| `code/exp3_detect.py`, `code/negatives.py` | per-dialogue screening with negative dialogues (earlier draft) |
| `code/exp3_table4.py` | message-level versus path-level audit; prepared but not run for the manuscript |
| `code/diag_elicit.py` | diagnostic for the induction prompts |
| `code/second_annotator.py`, `annotation2/` | first version of the second-annotator sheets; replaced by `code/make_annotation_xlsx.py` and `code/collect_annotations.py` |
| `code/README_ieee_access_draft.md` | documentation of an earlier draft (old table numbering) |
| `docs/` | the authors' internal notes on the code and datasets, translated into English from the Korean originals (which remain in the git history) |
| `dev_logs/` | pilot, smoke and debugging runs (console output recorded as is, partly in Korean; labels glossed in `docs/CODEBOOK.md`, section 5) |
| `results/exp3_stats_before_mr4_fix.json`, `results/logs/exp3_stats_run.txt` | paired tests before the MR4 correction |
| `results/logs/exp3_compare_output.txt`, `results/logs/exp3_detect_value_exposed.txt` | outputs of the archived scripts |
| `results/logs/exp1_run_2026-08-23.txt` | first run of the controlled check; MuSiQue used the first 500 questions instead of the 1,405 reported |
