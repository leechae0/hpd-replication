# data/

* `patil/` — the 119 Patil et al. scenario files (`experiment_<id>_def_adv_both.json`). Not stored in
  git; run `python3 code/fetch_patil.py`, which downloads them from
  https://github.com/Vaidehi99/MultiAgentPrivacy and verifies every file against `patil_manifest.json`.
* `patil_manifest.json` — file names and MD5 sums of the 119 files as used in the paper.
* `dialogues/patil_dialogues_116.jsonl` — the 116 rendered dialogues shown to the observers in
  Experiment 3 (value-preserving rendering; see `docs/DIALOGUE_RULES.md`). Regenerate with
  `python3 code/export_dialogues.py`.
* `hotpot_dev_distractor_v1.json`, `musique_ans_v1.0_dev.jsonl` — Experiment 1 only; produced by
  `python3 code/fetch_datasets.py`.
