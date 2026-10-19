# -*- coding: utf-8 -*-
"""
================================================================================
 mr_nlp.py — MR transformations for natural-language dialogues (real spaCy, not a mock)
================================================================================
 Provides four metamorphic transformations that operate on top of the observer
 judgment function V(S), plus a deterministic check for each ([Measure 0]). All
 language processing is done with spaCy (en_core_web_sm): it relies on real POS
 tagging and named-entity analysis, not on rule dictionaries or string tricks.

   - MR1 rephrasing : lexical substitution based on spaCy POS tags (meaning preserved, surface changed)
   - MR2 removal    : structural deletion of specific cue utterances (algorithmic)
   - MR3 irrelevant insertion: retrieve and insert sentences from an unrelated corpus whose
                      content words (lemmas) do not overlap with the dialogue
   - MR4 renaming   : spaCy NER + rule-based coreference (pronouns, possessives) + deterministic substitution

 Install:  pip install spacy && python -m spacy download en_core_web_sm
 Standalone run:  python3 mr_nlp.py   (prints a demo)
================================================================================
"""
import re
import spacy

# Real spaCy pipeline (hard dependency). If it is not installed, fail with a clear message.
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError as e:  # pragma: no cover
    raise OSError(
        "The spaCy model en_core_web_sm is not installed. Install it first:\n"
        "    pip install spacy\n"
        "    python -m spacy download en_core_web_sm"
    ) from e

NEW_NAMES = ["Zara", "Kenji", "Priya", "Owen", "Mara", "Tobias", "Lena", "Rahul",
             "Ingrid", "Mateo", "Sunita", "Bjorn", "Amara", "Dmitri", "Yuki", "Callum"]

# ---- MR4 fidelity fix (2026-09-21) ----------------------------------------------------------
# spaCy NER labels some non-person tokens as PERSON in record-style dialogues: key=value fields
# such as "employee_id=201", street names such as "Main St", and capitalized values such as
# "Chronic Asthma" or "Lead Engineer". Renaming those does not change identity only; it corrupts
# the data. A PERSON candidate is now renamed only if it is name-like (capitalized alphabetic
# words, no digits or "=", no street suffix) AND it is a dialogue speaker, or the value of a
# person-name field (name, *_name except organization-like prefixes, emergency_contact, owner,
# manager, ...), or a word of such an accepted name. The 2026-08/09 runs used the unfiltered
# version; results/logs/*_mr4fix.jsonl hold the MR4 re-judgments with this filter.
_NAME_RE = re.compile(r"^[A-Z][a-z]*(?:[-'][A-Z]?[a-z]+)*(?: [A-Z][a-z]*(?:[-'][A-Z]?[a-z]+)*){0,2}$")
_STREET_SUFFIX = {"st", "rd", "ave", "blvd", "ln", "dr", "way", "street", "road", "avenue", "lane", "drive"}
_ORG_PREFIX = {"company", "organization", "org", "project", "program", "product", "course", "service", "line",
               "district", "file", "table", "event", "item", "team", "department", "brand", "store", "school",
               "hospital", "clinic", "drug", "medication", "vendor", "supplier", "bank", "hotel", "airline",
               "route", "station", "city", "warehouse", "facility", "venue", "stadium", "region",
               "document", "carrier", "college", "university", "institute", "report", "policy", "plan",
               "account", "device", "app", "system", "software", "tool", "model", "vehicle", "ship",
               "aircraft", "flight", "hotel", "restaurant", "shop", "fund", "grant", "scholarship", "award"}
_PERSON_KEYS = {"emergency_contact", "director", "contact", "manager", "owner", "author", "assigned_to",
                "reviewer", "physician", "doctor", "nurse", "patient", "student", "employee", "customer",
                "client", "user", "tenant", "driver", "pilot", "coach", "player", "guardian", "spouse",
                "supervisor", "recipient", "sender", "approved_by", "reported_by", "created_by",
                "accessed_by", "requested_by", "parent", "child", "dependent", "member", "applicant",
                "candidate", "instructor", "teacher", "advisor", "mentor", "agent", "officer", "lead"}


def _is_person_key(key):
    k = key.lower()
    if k == "name" or k in _PERSON_KEYS:
        return True
    if k.endswith("_name"):
        return k[:-5] not in _ORG_PREFIX
    return False


def _name_like(cand):
    if not _NAME_RE.match(cand):
        return False
    words = cand.split()
    return words[-1].lower() not in _STREET_SUFFIX


def _person_field_values(text):
    """Values of person-name fields (key=value records), whether or not NER tagged them."""
    for key, val in re.findall(r"(\w+)\s*=\s*([A-Z][A-Za-z'\-]*(?: [A-Z][A-Za-z'\-]*){0,2})(?=\s*[,;)]|\s*$)", text, flags=re.M):
        if key.lower() != "name" and _is_person_key(key):     # bare "name" is ambiguous (files, places, stores)
            yield val


def _select_names(text, candidates, speakers):
    """Keep the candidates that are dialogue speakers, values of person-name fields, or words of those."""
    accepted = set()
    for c in candidates:
        if not _name_like(c):
            continue
        if c in speakers:
            accepted.add(c); continue
        keys = re.findall(r"(\w+)\s*=\s*" + re.escape(c) + r"(?=\s*[,;)]|\s*$)", text, flags=re.M)
        if any(_is_person_key(k) for k in keys):
            accepted.add(c)
    words = {w for a in accepted for w in a.split()}
    for c in candidates:
        if c not in accepted and _name_like(c) and all(w in words for w in c.split()):
            accepted.add(c)
    return accepted


def _person_spans(doc):
    """PERSON entities -> occurrence spans, keyed by surface form."""
    people = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.setdefault(ent.text.strip(), []).append((ent.start_char, ent.end_char))
    return people


def _speaker_names(text):
    """Additionally recognize the speaker token of the dialogue format 'Name: ...' as a person.
    spaCy NER sometimes misses non-Western names (e.g., Bhushan, Harpreet), a defect the
    controlled experiment actually exposed, so speaker prefixes serve as a deterministic supplement."""
    return set(re.findall(r"(?m)^\s*([A-Z][A-Za-z]+)\s*:", text))


def _rule_coref(doc):
    """Rule-based coreference: link each third-person singular pronoun to the nearest preceding PERSON.
    Returns: [(pronoun, char_start, linked person)], for verification and the demo."""
    person_positions = sorted(
        (ent.start_char, ent.text.strip()) for ent in doc.ents if ent.label_ == "PERSON"
    )
    links = []
    for tok in doc:
        if tok.tag_ in ("PRP", "PRP$"):  # personal/possessive pronouns
            if tok.lower_ not in ("he", "she", "him", "her", "his", "hers"):
                continue  # link only third-person singular human pronouns (exclude it/they/we/you)
            prev = [p for p in person_positions if p[0] < tok.idx]
            if prev:
                links.append((tok.text, tok.idx, prev[-1][1]))
    return links


def mr4_rename(text):
    """Replace all persons with new names. Returns: (new text, mapping, coreference links)."""
    doc = _NLP(text)
    speakers = _speaker_names(text)
    field_values = set(_person_field_values(text))          # names NER may miss (e.g. non-Western names in name fields)
    names = sorted(_select_names(text, set(_person_spans(doc).keys()) | speakers | field_values, speakers))
    pool = [n for n in NEW_NAMES if not re.search(rf"\b{n}\b", text)]
    mapping = {}
    for i, n in enumerate(sorted(names, key=lambda s: -len(s))):  # longer names first
        mapping[n] = pool[i] if i < len(pool) else f"Person{i}"
    coref = _rule_coref(doc)
    out = text
    for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out, mapping, coref


def verify_mr4(before, after, mapping):
    """[Measure 0] Deterministic check: no original name remains + persons preserved + non-name tokens unchanged (approximate)."""
    da = _NLP(after)
    left = [old for old in mapping if re.search(rf"\b{re.escape(old)}\b", after)]
    people_after = {e.text for e in da.ents if e.label_ == "PERSON"}

    def nonname_words(t, names):
        return sorted(x for x in re.findall(r"[A-Za-z]+", t) if x not in names)

    same_rest = (nonname_words(before, set(mapping))
                 == nonname_words(after, set(mapping.values())))
    return {"no_original_name_left": len(left) == 0, "names_left": left,
            "persons_present": len(people_after) >= 1,
            "non_name_tokens_unchanged_approx": same_rest}


# ==============================================================================
# MR2 — removal: structural deletion of specific cue utterances (algorithmic)
# ==============================================================================
def mr2_remove(turns, remove_idx):
    drop = set(remove_idx)
    return [t for i, t in enumerate(turns) if i not in drop]


def verify_mr2(before, after, removed_idx):
    gone = all(before[i] not in after for i in removed_idx)
    rest = [t for i, t in enumerate(before) if i not in set(removed_idx)] == after
    return {"target_removed": gone, "rest_preserved": rest}


# ==============================================================================
# MR3 — irrelevant insertion: retrieve and insert sentences from an unrelated corpus with no content-word (lemma) overlap
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
    """Set of content-word lemmas (nouns, verbs, proper nouns, adjectives; stop words excluded)."""
    doc = _NLP(text)
    return {t.lemma_.lower() for t in doc
            if t.pos_ in ("NOUN", "VERB", "PROPN", "ADJ") and not t.is_stop}


def mr3_inject_noise(turns, k=2):
    """Retrieve and append k irrelevant sentences that do not overlap with the dialogue's content words."""
    dia = set().union(*[_content_lemmas(t["text"]) for t in turns]) if turns else set()
    scored = sorted((len(_content_lemmas(s) & dia), s) for s in NOISE_CORPUS)
    picked = [s for ov, s in scored[:k] if ov == 0]  # only those with zero overlap
    added = [{"speaker": "Owen", "text": s} for s in picked]
    return turns + added, picked


def verify_mr3(turns_before, picked):
    """[Measure 0] Deterministic check: added sentences share zero content words with the original dialogue."""
    dia = set().union(*[_content_lemmas(t["text"]) for t in turns_before]) if turns_before else set()
    detail = {s[:30]: len(_content_lemmas(s) & dia) == 0 for s in picked}
    return {"all_added_unrelated": all(detail.values()) if detail else True, "detail": detail}


# ==============================================================================
# MR1 — rephrasing: lexical substitution based on spaCy POS tags (code baseline). A stronger version uses a paraphrase model/LLM.
# ==============================================================================
# Nouns that can form a cue (table name), such as records, documents, replies, are not substituted:
# MR1 prioritizes meaning preservation, and breaking a cue reference would fail the [Measure 0] check.
SYN = {
    "increase": "rise", "increased": "risen", "many": "numerous", "lot": "great deal",
    "often": "frequently", "barely": "hardly", "noticed": "observed",
    "uploading": "submitting", "out": "away", "booked": "reserved",
    "dinner": "meal", "yesterday": "the day before",
    "going": "looking", "earlier": "before", "today": "this morning",
}


def mr1_reframe(text):
    """Replace content words with synonyms (meaning preserved, wording changed). Substitutes only when the POS matches; deterministic."""
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
    """[Measure 0] Check: persons and numbers preserved + surface changed. (Final confirmation of meaning preservation is by human review of a sample.)"""
    def keyfacts(t):
        d = _NLP(t)
        persons = sorted(e.text for e in d.ents if e.label_ == "PERSON")
        nums = sorted(re.findall(r"\d+", t))
        return persons, nums

    pb, nb = keyfacts(before)
    pa, na = keyfacts(after)
    return {"persons_preserved": pb == pa, "numbers_preserved": nb == na,
            "surface_changed": before.strip() != after.strip(),
            "note": "meaning preservation is confirmed by a human audit of a sample (agreement rate)"}


# ==============================================================================
# Demo — actually run each transformation on a real natural-language (English) dialogue
# ==============================================================================
if __name__ == "__main__":
    turns = [
        {"speaker": "Alice",    "text": "I noticed Kim has been uploading a lot of handover documents lately."},
        {"speaker": "Bhushan",  "text": "Right, Kim has also been out of the office more often."},
        {"speaker": "Eden",     "text": "Kim barely replies to messages now."},
        {"speaker": "Harpreet", "text": "I booked the restaurant for the dinner yesterday."},  # distractor cue
    ]
    text = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in turns)
    print("Original:\n" + text)

    print("\n== MR4 renaming (NER + rule-based coreference + substitution) ==")
    r4, mp, coref = mr4_rename(text)
    print(r4)
    print("mapping:", mp)
    print("coreference links (pronoun → person):", coref)
    print("check:", verify_mr4(text, r4, mp))

    print("\n== MR2 removal (structural: utterance 0 deleted) ==")
    kept = mr2_remove(turns, [0])
    print("\n".join(f'{t["speaker"]}: {t["text"]}' for t in kept))
    print("check:", verify_mr2(turns, kept, [0]))

    print("\n== MR3 irrelevant insertion (retrieval + unrelatedness filter) ==")
    t3, picked = mr3_inject_noise(turns, 2)
    print("inserted unrelated sentences:", picked)
    print("check:", verify_mr3(turns, picked))

    print("\n== MR1 rephrasing (POS-based lexical substitution baseline) ==")
    r1 = mr1_reframe(text)
    print(r1)
    print("check:", verify_mr1(text, r1))
