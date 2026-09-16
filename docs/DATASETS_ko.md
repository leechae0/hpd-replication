# 데이터셋 상세

각 실험이 데이터에 요구하는 것이 다르다. 정량 채점에는 **"어떤 단서 조합이 비밀을 만드는가"의
정답 레이블**이 필요하며, 이 레이블이 있는 데이터에서만 자카드·soundness·coverage를 보고한다.

| 데이터셋 | 정답 레이블(gt) | 규모 | 사용 실험 | 다운로드 |
|---|---|---|---|---|
| **Patil** | `defense` = 민감 조합(최소 지지집합) | 시나리오 116, 단서 ~8 | 1·2·3 | github.com/Vaidehi99/MultiAgentPrivacy (repo 내 `data/`) |
| **HotpotQA** | `supporting_facts`의 gold 문단 | 문항 ~7.4k, 문단 10 | 1 | hotpotqa.github.io (dev distractor json) |
| **MuSiQue** | `is_supporting == true` 문단 | 문항 ~2.5k, 문단 ~20 | 1·3(확장) | github.com/StonyBrookNLP/musique (`download_data.sh`) |
| **MAGPIE** | 없음(정성 시연) | 자유형 대화 200 | 3(시연) | huggingface.co/datasets/jaypasnagasai/magpie |

## 1. Patil MultiAgentPrivacy — 정량 코어
- **구조**: 시나리오마다 참여자 4명이 테이블(=단서)을 나눠 보유. 전체 단서 ~8개.
  `defense` 필드가 "이 테이블 조합이 민감"을 명시 → 그 테이블 집합이 **최소 지지집합(정답)**.
  나머지 ~3개는 교란 단서. `run_2_sensitive.final_inference_result`가 비밀 z.
- **로더**: `datasets.load_patil()`. 정답 색인 `gt`를 자동 산출. 단서 텍스트는 실 spaCy 변형이
  작동하도록 영어로 렌더한다(원 데이터가 영어).
- **역할**: 정답 레이블이 있는 유일한 프라이버시 데이터 → 세 실험 모두의 정량 기준.

## 2. HotpotQA (distractor) — 규모·제2 도메인
- **구조**: `context` = 문단 10개([제목, 문장들]). `supporting_facts` = [제목, 문장색인].
  supporting_facts에 제목이 등장하는 **gold 문단 2개 = 최소 지지집합**, 나머지 8개 = 교란.
  비밀 z = 다중홉 질문의 답(단일 문단으로는 안 나옴).
- **로더**: `datasets.load_hotpotqa(path)`.
- **역할**: 실험 1의 규모·일반성(프라이버시 밖에서도 탐색이 작동).

## 3. MuSiQue-Answerable — 규모 확장·확장 곡선
- **구조**: `paragraphs`(각 `is_supporting` 플래그, ~20개), `question_decomposition`(홉별 분해).
  is_supporting=true 문단(2~4개) = 최소 지지집합. 나머지 ~16개 = 교란.
- **로더**: `datasets.load_musique(path, min_hops, max_hops, limit)`.
- **역할**: 단서 규모(~20)와 홉 다양성으로 확장 곡선. 실험 1·3.
- **특징**: "끊긴 추론"을 막도록 설계돼 모든 홉을 실제로 거쳐야 답이 나옴 → 조합 추론 취지에 부합.

## 4. MAGPIE — 자유형 정성 시연
- **구조**: 다중 에이전트 자유형 협상 대화. 경로 레이블 없음.
- **로더**: `datasets.load_magpie(path, limit)` (gt 비어 있음).
- **역할**: 실험 3에서 레이블 없는 실제 대화에도 방법이 도는지 **정성 시연**. 정량 성능은 주장하지 않음.

## 실험별 데이터 매핑 (요약)
- **실험 1(탐색)**: Patil + HotpotQA + MuSiQue — 정확성·일반성·확장성.
- **실험 2(인증)**: Patil 단일 — MR 변형이 표 형태 단서에서 신뢰성 있게 적용되므로.
- **실험 3(종단)**: Patil(대화 변환, 정량) + MuSiQue(규모) + MAGPIE(자유형 시연).

## 다운로드가 이 환경에서 막힌 이유
HuggingFace·Google Drive·zenodo 가 이 샌드박스에서 차단되어 HotpotQA·MuSiQue·MAGPIE 원본은
여기서 받을 수 없다. 로더는 **실제 파일 포맷을 그대로 파싱**하므로, 위 출처에서 파일을 받아
경로만 지정하면 즉시 동작한다. Patil은 repo에 포함되어 있어 바로 사용 가능하다.
