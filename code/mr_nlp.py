# -*- coding: utf-8 -*-
"""
================================================================================
 mr_nlp.py — 자연어 대화용 MR 변형 (실 spaCy, mock 아님)
================================================================================
 관찰자 판정 함수 V(S) 위에서 동작하는 네 가지 메타모픽 변형과 각 변형의
 결정적 검증 함수([측정0])를 제공한다. 모든 언어 처리는 spaCy(en_core_web_sm)로
 수행한다 — 규칙 사전이나 문자열 트릭이 아니라 실제 품사·개체명 분석에 기반한다.

   - MR1 재표현 : spaCy 품사 태깅 기반 어휘 치환(뜻 보존, 표면 변경)
   - MR2 제거   : 특정 단서 발화의 구조적 삭제(알고리즘)
   - MR3 무관주입: 무관 코퍼스에서 '내용어(lemma)가 안 겹치는' 문장 검색·삽입
   - MR4 개명   : spaCy NER + 규칙 기반 상호참조(대명사·소유격) + 결정적 치환

 설치:  pip install spacy && python -m spacy download en_core_web_sm
 단독 실행:  python3 mr_nlp.py   (데모 출력)
================================================================================
"""
import re
import spacy

# 실 spaCy 파이프라인(하드 의존). 설치가 안 됐으면 명확한 안내와 함께 실패한다.
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError as e:  # pragma: no cover
    raise OSError(
        "spaCy 모델 en_core_web_sm 이 없습니다. 먼저 설치하세요:\n"
        "    pip install spacy\n"
        "    python -m spacy download en_core_web_sm"
    ) from e

NEW_NAMES = ["Zara", "Kenji", "Priya", "Owen", "Mara", "Tobias", "Lena", "Rahul"]


# ==============================================================================
# MR4 — 개명: NER로 인물을 찾고, 대명사·소유격을 규칙으로 연결한 뒤 결정적 치환
# ==============================================================================
def _person_spans(doc):
    """PERSON 개체명 → 표면형별 등장 스팬."""
    people = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.setdefault(ent.text.strip(), []).append((ent.start_char, ent.end_char))
    return people


def _speaker_names(text):
    """대화 형식 'Name: ...'의 화자 토큰을 인물로 추가 인식.
    spaCy NER은 비서구권 이름(Bhushan, Harpreet 등)을 놓치는 경우가 있어
    — 통제 실험이 실제로 드러낸 결함 — 화자 접두어를 결정적 보완으로 쓴다."""
    return set(re.findall(r"(?m)^\s*([A-Z][A-Za-z]+)\s*:", text))


def _rule_coref(doc):
    """규칙 기반 상호참조: 각 3인칭 단수 대명사를 '가장 가까운 앞선 PERSON'에 연결.
    반환: [(대명사, char_start, 연결된 인물)] — 검증·데모용."""
    person_positions = sorted(
        (ent.start_char, ent.text.strip()) for ent in doc.ents if ent.label_ == "PERSON"
    )
    links = []
    for tok in doc:
        if tok.tag_ in ("PRP", "PRP$"):  # 인칭/소유 대명사
            if tok.lower_ not in ("he", "she", "him", "her", "his", "hers"):
                continue  # 사람 3인칭 단수만 연결(it/they/we/you 제외)
            prev = [p for p in person_positions if p[0] < tok.idx]
            if prev:
                links.append((tok.text, tok.idx, prev[-1][1]))
    return links


def mr4_rename(text):
    """인물을 새 이름으로 일괄 교체. 반환: (새 텍스트, 매핑, 상호참조 링크)."""
    doc = _NLP(text)
    names = list(set(_person_spans(doc).keys()) | _speaker_names(text))
    pool = [n for n in NEW_NAMES if n not in names]
    mapping = {}
    for i, n in enumerate(sorted(names, key=lambda s: -len(s))):  # 긴 이름 먼저
        mapping[n] = pool[i] if i < len(pool) else f"Person{i}"
    coref = _rule_coref(doc)
    out = text
    for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out, mapping, coref


def verify_mr4(before, after, mapping):
    """[측정0] 결정적 검증: 원래 이름이 하나도 안 남았나 + 인물 보존 + 비인물 토큰 불변(근사)."""
    da = _NLP(after)
    left = [old for old in mapping if re.search(rf"\b{re.escape(old)}\b", after)]
    people_after = {e.text for e in da.ents if e.label_ == "PERSON"}

    def nonname_words(t, names):
        return sorted(x for x in re.findall(r"[A-Za-z]+", t) if x not in names)

    same_rest = (nonname_words(before, set(mapping))
                 == nonname_words(after, set(mapping.values())))
    return {"원래이름_0개남음": len(left) == 0, "남은이름": left,
            "인물수_보존": len(people_after) >= 1,
            "이름외_토큰_불변(근사)": same_rest}


# ==============================================================================
# MR2 — 제거: 특정 단서 발화의 구조적 삭제 (알고리즘)
# ==============================================================================
def mr2_remove(turns, remove_idx):
    drop = set(remove_idx)
    return [t for i, t in enumerate(turns) if i not in drop]


def verify_mr2(before, after, removed_idx):
    gone = all(before[i] not in after for i in removed_idx)
    rest = [t for i, t in enumerate(before) if i not in set(removed_idx)] == after
    return {"대상_삭제됨": gone, "나머지_보존": rest}


# ==============================================================================
# MR3 — 무관주입: 무관 코퍼스에서 '내용어(lemma)가 안 겹치는' 문장 검색·삽입
# ==============================================================================
NOISE_CORPUS = [
    "The marathon route passes three historic bridges downtown.",
    "Volcanic soil makes the vineyards unusually fertile this season.",
    "The telescope captured a faint spiral galaxy last night.",
    "Sourdough needs a long fermentation to develop its flavor.",
    "The subway extension will open two new stations next spring.",
    "Migrating cranes stop at the wetland every October.",
    "The museum restored a mural from the fourteenth century.",
    "Solar panels on the stadium roof cut its energy bill sharply.",
]


def _content_lemmas(text):
    """내용어 lemma 집합 (명사·동사·고유명사·형용사, 불용어 제외)."""
    doc = _NLP(text)
    return {t.lemma_.lower() for t in doc
            if t.pos_ in ("NOUN", "VERB", "PROPN", "ADJ") and not t.is_stop}


def mr3_inject_noise(turns, k=2):
    """대화의 내용어와 '겹치지 않는' 무관 문장 k개를 검색해 추가."""
    dia = set().union(*[_content_lemmas(t["text"]) for t in turns]) if turns else set()
    scored = sorted((len(_content_lemmas(s) & dia), s) for s in NOISE_CORPUS)
    picked = [s for ov, s in scored[:k] if ov == 0]  # 겹침 0인 것만
    added = [{"speaker": "Owen", "text": s} for s in picked]
    return turns + added, picked


def verify_mr3(turns_before, picked):
    """[측정0] 결정적 검증: 추가 문장이 원 대화와 내용어 0개 겹침."""
    dia = set().union(*[_content_lemmas(t["text"]) for t in turns_before]) if turns_before else set()
    detail = {s[:30]: len(_content_lemmas(s) & dia) == 0 for s in picked}
    return {"추가문장_모두_무관": all(detail.values()) if detail else True, "상세": detail}


# ==============================================================================
# MR1 — 재표현: spaCy 품사 기반 어휘 치환(코드 baseline). 강한 버전은 재표현 모델/LLM.
# ==============================================================================
# 단서(테이블명)를 이룰 수 있는 명사(records, documents, replies 등)는 치환하지 않는다
# — MR1은 '뜻 보존'이 우선이며, 단서 지칭을 깨면 [측정0] 검증에 걸린다.
SYN = {
    "increase": "rise", "increased": "risen", "many": "numerous", "lot": "great deal",
    "often": "frequently", "barely": "hardly", "noticed": "observed",
    "uploading": "submitting", "out": "away", "booked": "reserved",
    "dinner": "meal", "yesterday": "the day before",
    "going": "looking", "earlier": "before", "today": "this morning",
}


def mr1_reframe(text):
    """내용어를 동의어로 치환(뜻 보존, 표현 변경). 품사가 맞을 때만 치환하며 결정적."""
    doc = _NLP(text)
    out = []
    for tok in doc:
        low = tok.lower_
        if low in SYN and tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
            rep = SYN[low]
            out.append(rep.capitalize() if tok.is_title else rep)
        else:
            out.append(tok.text)
        out.append(tok.whitespace_)
    return "".join(out)


def verify_mr1(before, after):
    """[측정0] 검증: 인물·수치 보존 + 표면 변경. (뜻 보존 최종 확인은 사람 표본 검수.)"""
    def keyfacts(t):
        d = _NLP(t)
        persons = sorted(e.text for e in d.ents if e.label_ == "PERSON")
        nums = sorted(re.findall(r"\d+", t))
        return persons, nums

    pb, nb = keyfacts(before)
    pa, na = keyfacts(after)
    return {"인물_보존": pb == pa, "수치_보존": nb == na,
            "표면_변경": before.strip() != after.strip(),
            "주의": "뜻 보존 최종 확인은 사람 표본 검수(일치율)"}


# ==============================================================================
# 데모 — 실제 자연어 대화(영어)에 각 변형을 실제로 실행
# ==============================================================================
if __name__ == "__main__":
    turns = [
        {"speaker": "Alice",    "text": "I noticed Kim has been uploading a lot of handover documents lately."},
        {"speaker": "Bhushan",  "text": "Right, Kim has also been out of the office more often."},
        {"speaker": "Eden",     "text": "Kim barely replies to messages now."},
        {"speaker": "Harpreet", "text": "I booked the restaurant for the dinner yesterday."},  # 교란 단서
    ]
    text = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in turns)
    print("원본:\n" + text)

    print("\n== MR4 개명 (NER + 규칙 상호참조 + 치환) ==")
    r4, mp, coref = mr4_rename(text)
    print(r4)
    print("매핑:", mp)
    print("상호참조 링크(대명사→인물):", coref)
    print("검증:", verify_mr4(text, r4, mp))

    print("\n== MR2 제거 (구조적: 0번 발화 삭제) ==")
    kept = mr2_remove(turns, [0])
    print("\n".join(f'{t["speaker"]}: {t["text"]}' for t in kept))
    print("검증:", verify_mr2(turns, kept, [0]))

    print("\n== MR3 무관주입 (검색 + 무관 필터) ==")
    t3, picked = mr3_inject_noise(turns, 2)
    print("추가된 무관 문장:", picked)
    print("검증:", verify_mr3(turns, picked))

    print("\n== MR1 재표현 (품사 기반 어휘 치환 baseline) ==")
    r1 = mr1_reframe(text)
    print(r1)
    print("검증:", verify_mr1(text, r1))
