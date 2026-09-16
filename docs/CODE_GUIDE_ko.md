# HPD 실험 코드 설명 (실코드, mock 없음)

관찰자는 **실제 LLM**, MR 변형은 **실제 자연어 처리(spaCy)**, 데이터는 **실 파일 파서**로 동작한다.
실험 1·2는 LLM 없이(무료·오프라인) 돌고, 실험 3은 LLM 키가 필요하다.

## 디렉터리 구조
```
hpd/
├─ llm.py            실 LLM 백엔드(Groq/OpenAI/Anthropic) + Observer(judge/elicit/explain, r-vote)
├─ datasets.py       실 데이터 로더: Patil·HotpotQA·MuSiQue·MAGPIE + Patil→대화 변환(실 LLM)
├─ mr_nlp.py         실 MR 변형: MR4=NER+상호참조+치환 · MR3=검색+무관필터 · MR2=구조삭제 · MR1=어휘치환
├─ search.py         탐색 알고리즘: backward_search(최소집합 1개) · enumerate_all(완전열거) · jaccard
├─ negatives.py      대화 수준 탐지용 음성(비누출) 시나리오 3종 생성
├─ exp1_search.py    ▶ 실험 1 — 탐색 정확성·강건성·확장성 (통제 오라클)
├─ exp2_certify.py   ▶ 실험 2 — 메타모픽 인증 타당성 (실 spaCy MR)
├─ exp3_endtoend.py  ▶ 실험 3-1 — 실제 LLM 종단 (유도→탐색→인증 + 베이스라인·단조성·실가짜 coverage)
└─ exp3_detect.py    ▶ 실험 3-2 — 대화 수준 누출 탐지 (양성+음성 풀, precision/recall/F1/오탐률)
```

## 모듈별 역할

**llm.py** — `make_backend("groq"|"openai"|"anthropic")` 로 실 백엔드 생성. `Observer.judge_excerpt`(발췌로 z 결론 가능?), `elicit`(열린 유도, 4프레이밍 합집합), `explain`(자기설명 베이스라인). 모든 판정은 r회 다수결. mock 백엔드 없음.

**datasets.py** — 4개 데이터셋을 공통 `Scenario`(units·gt·secret)로 통일. `gt`(정답 최소집합)는 Patil(defense)·HotpotQA(supporting_facts)·MuSiQue(is_supporting)에서만 존재하고 MAGPIE는 비어 있다(정성 시연). `scenario_to_dialogue(backend, s)`는 Patil 시나리오를 실 LLM으로 자연어 대화(단서 1개=발화 1개, 라벨 보존)로 변환.

**mr_nlp.py** — 네 변형과 각 변형의 결정적 검증 함수(`verify_mr1/3/4`). MR4는 spaCy NER과 화자 접두어로 인물을 찾아 대명사·소유격까지 규칙 기반으로 연결한 뒤 결정적으로 치환한다. MR3은 무관 코퍼스에서 내용어가 겹치지 않는 문장을 검색해 삽입한다. MR1은 어휘 치환(baseline)이며 강한 버전은 재표현 모델/LLM로 교체 가능하다.

**search.py** — 관찰자 판정 함수 `V(S)->{0,1}` 위에서 동작. `backward_search`는 전체에서 각 단서를 한 번씩 제거해 최소 지지집합 1개를 |U|+1 질의로 복원(=MR2). `enumerate_all`은 모든 최소집합을 완전 열거(소규모용).

## 실행

### 실험 1 (무료·오프라인)
```bash
python3 exp1_search.py
# 실제 QA 파일이 있으면 함께:
python3 exp1_search.py --hotpotqa data/hotpot_dev_distractor_v1.json \
                       --musique  data/musique_ans_v1.0_dev.jsonl
```
관찰자는 정답 기반 **통제 오라클**(LLM 아님). 잡음 ε(0/0.05/0.10/0.20)를 주입하고 r(1/5/7/9)로 다수결하며 5시드 평균±표준편차, 무작위·전수 베이스라인을 함께 보고.

### 실험 2 (무료·오프라인, spaCy 필요)
```bash
pip install spacy && python -m spacy download en_core_web_sm
python3 exp2_certify.py
```
변형은 실 spaCy로 수행. 통제 관찰자(진성 1 + 스퓨리어스 4종)로 [0]구현검증·[1]soundness·[2]coverage·[3]비중복성 산출.

### 실험 3 (실 LLM)
```bash
pip install openai spacy && python -m spacy download en_core_web_sm
export GROQ_API_KEY=<키>
python3 exp3_endtoend.py --backend groq --limit 20 --r 3     # 3-1 파일럿
python3 exp3_endtoend.py --backend groq --r 5                # 3-1 전량
python3 exp3_endtoend.py --backend openai --model gpt-4o-2024-08-06   # 3-1 프론티어
```
(3-1) 유도 재현율(A/B/C 게이트), 탐색 자카드, 실측 soundness, 단조성 위반율, 베이스라인 2종, 실가짜 coverage 보고.

```bash
# (3-2) 대화 수준 탐지 — 양성(Patil)+음성(비누출) 풀에서 누출 대화 골라내기
python3 exp3_detect.py --backend groq --limit 20 --r 3
python3 exp3_detect.py --backend groq --limit 20 --r 3 --magpie data/magpie.jsonl  # 자연 정상 대화 추가
```
(3-2) 음성은 negatives.py로 구성(민감 조합만 제거 / 단서 뒤섞기 / 자연 정상). 각 대화에 유도→탐색→인증을 적용해 인증 경로가 남으면 누출로 플래그하고, 대화 수준 정밀도·재현율·F1·오탐률과 양성 적중 시 기밀 일치도·경로 자카드를 보고.

## 오프라인 실측(현재 코드)
- 실험 1(Patil): 무잡음 116/116, 강잡음(ε=0.20) 정확복원 96.4±3.8·자카드 0.927, 질의 9 vs 전수 256, 무작위 0.431.
- 실험 2(실 spaCy): [0] MR4·MR3 116/116·MR1 111/116, [1] soundness 1.000, [2] coverage 75.0%(cooccur 미검출), [3] 비중복성 각 116/116.
- 실험 3: 실 LLM 키 필요(유도 재현율이 게이트 지표).

## 주의: '통제 오라클'과 '통제 관찰자'는 mock이 아니다
실험 1의 오라클과 실험 2의 진성/스퓨리어스 관찰자는 정답을 아는 **통제 환경의 실험 장치**다(정답과 대조, 진위 주입). 이는 방법 검증을 위한 설계이며, 변형 자체는 실 NLP 코드다. 실제 LLM 관찰자는 실험 3에서 사용한다.
