# Datasets: internal notes (earlier draft)

English translation of the authors' internal notes, originally written in Korean
(`archive/docs_ko/DATASETS_ko.md` in earlier releases). They describe an earlier plan and are kept
for transparency only. The PeerJ manuscript uses the Patil benchmark for every observer experiment
and HotpotQA and MuSiQue only for the controlled checks (Section S5); MAGPIE is not used. See the
top-level `README.md` and `data/README.md` for the current data.

Each experiment needs something different from the data. Quantitative scoring needs **annotations
of which cue combination produces the secret**; Jaccard, soundness and coverage are reported only
on data that have them.

| Dataset | Annotation (gt) | Size | Experiments | Download |
|---|---|---|---|---|
| **Patil** | `defense` = sensitive combination (minimal supporting set) | 116 scenarios, ~8 cues | 1, 2, 3 | github.com/Vaidehi99/MultiAgentPrivacy (`data/` in the repository) |
| **HotpotQA** | gold paragraphs of `supporting_facts` | ~7.4k questions, 10 paragraphs | 1 | hotpotqa.github.io (dev distractor json) |
| **MuSiQue** | paragraphs with `is_supporting == true` | ~2.5k questions, ~20 paragraphs | 1, 3 (extension) | github.com/StonyBrookNLP/musique (`download_data.sh`) |
| **MAGPIE** | none (qualitative demonstration) | 200 free-form dialogues | 3 (demonstration) | huggingface.co/datasets/jaypasnagasai/magpie |

## 1. Patil MultiAgentPrivacy — quantitative core
- **Structure**: in each scenario four participants hold tables (= cues), about 8 cues in all. The
  `defense` field states which combination of tables is sensitive → that set of tables is the
  **minimal supporting set (annotation)**. The remaining ~3 are distractor cues.
  `run_2_sensitive.final_inference_result` is the secret z.
- **Loader**: `datasets.load_patil()`, which derives the annotated indices `gt`. Cue texts are
  rendered in English so that the real spaCy transformations work (the original data are English).
- **Role**: the only privacy dataset with annotations → the quantitative reference for all three
  experiments.

## 2. HotpotQA (distractor) — scale and a second domain
- **Structure**: `context` = 10 paragraphs ([title, sentences]). `supporting_facts` = [title,
  sentence index]. The **2 gold paragraphs** whose titles appear in supporting_facts = minimal
  supporting set; the other 8 = distractors. Secret z = the answer to the multi-hop question (not
  obtainable from a single paragraph).
- **Loader**: `datasets.load_hotpotqa(path)`.
- **Role**: scale and generality for Experiment 1 (search works outside privacy too).

## 3. MuSiQue-Answerable — scale extension and scaling curve
- **Structure**: `paragraphs` (each with an `is_supporting` flag, ~20), `question_decomposition`
  (per-hop decomposition). Paragraphs with is_supporting = true (2 to 4) = minimal supporting set;
  the other ~16 = distractors.
- **Loader**: `datasets.load_musique(path, min_hops, max_hops, limit)`.
- **Role**: scaling curve over cue count (~20) and hop diversity. Experiments 1 and 3.
- **Property**: designed to prevent disconnected reasoning, so every hop must actually be used to
  reach the answer → fits the idea of compositional inference.

## 4. MAGPIE — free-form qualitative demonstration
- **Structure**: multi-agent free-form negotiation dialogues. No path annotations.
- **Loader**: `datasets.load_magpie(path, limit)` (gt empty).
- **Role**: a **qualitative demonstration** in Experiment 3 that the method runs on unlabelled real
  dialogues. No quantitative performance is claimed.

## Data per experiment (summary)
- **Experiment 1 (search)**: Patil + HotpotQA + MuSiQue — accuracy, generality, scalability.
- **Experiment 2 (validation)**: Patil only — the MR transformations apply reliably to tabular cues.
- **Experiment 3 (end-to-end)**: Patil (converted to dialogue, quantitative) + MuSiQue (scale) +
  MAGPIE (free-form demonstration).

## Why downloads were blocked in the development environment
HuggingFace, Google Drive and Zenodo were blocked in the development sandbox, so the original
HotpotQA, MuSiQue and MAGPIE files could not be fetched there. The loaders **parse the real file
formats as is**, so they work once the files are downloaded from the sources above and the paths
are given. Patil was included in the repository and could be used directly.
