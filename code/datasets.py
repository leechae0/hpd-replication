# -*- coding: utf-8 -*-
"""
datasets.py — real dataset loaders (no mocks).
Unifies the 4 datasets under the common schema Scenario.

  Scenario.units  : list of cues. Each cue = {"agent","key","text"}
  Scenario.gt     : ground-truth minimal cue set (frozenset of cue indices). Empty if unavailable.
  Scenario.secret : secret z (string)

Quantitative scoring is possible only on data that has labels (gt):
  - Patil       : defense field = sensitive combination = minimal support set  (has gt)
  - HotpotQA    : gold paragraphs in supporting_facts                          (has gt)
  - MuSiQue     : paragraphs with is_supporting == true                        (has gt)
  - MAGPIE      : no path labels -> qualitative demonstration only              (no gt)
"""
import json, glob, os, re
from dataclasses import dataclass, field


@dataclass
class Scenario:
    sid: str
    units: list            # [{"agent","key","text"}]
    secret: str
    gt: frozenset = field(default_factory=frozenset)

    @property
    def entities(self):
        return sorted({u["agent"] for u in self.units})


# ------------------------------------------------------------ Patil
# Automatic data-path lookup: env var PATIL_DIR > data/ bundled with the project > default path.
def _find_patil_dir():
    if os.environ.get("PATIL_DIR"):
        return os.environ["PATIL_DIR"]
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, "..", "data", "patil"),   # replication package layout (data/patil/)
                 os.path.join(here, "..", "data"),
                 os.path.join(here, "data")]:
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "experiment_*.json")):
            return os.path.abspath(cand)
    return os.path.join(here, "..", "data", "patil")   # default location; run fetch_patil.py first

PATIL_DIR = _find_patil_dir()

def _defense_tables(text):
    m = re.search(r'combination of (.+?) is sensitive', text, re.I)
    if not m: return []
    return [t.strip() for t in m.group(1).replace(' and ', ', ').split(',') if t.strip()]

def _fmt_rows(rows, limit=6):
    """Render the actual values of the records in human-readable form (keeps the values the observer joins on)."""
    out = []
    for row in rows[:limit]:
        out.append("(" + ", ".join(f"{k}={v}" for k, v in row.items()) + ")")
    return "; ".join(out)

def load_patil(path=None):
    """Loader for the actual Patil MultiAgentPrivacy files. units = per-agent tables (cues)."""
    path = path or PATIL_DIR
    out = []
    for fp in sorted(glob.glob(os.path.join(path, "experiment_*_def_adv_both.json"))):
        d = json.load(open(fp))
        run = d.get("run_2_sensitive") or {}
        secret = run.get("final_inference_result")
        # Restrict the secret to the 'fact conjunction' part (the trailing speculation 'suggesting…' interprets beyond the data, so it is excluded).
        # The substance of compositional leakage is the fact conjunction determined by the cues; this is also all the observer can establish.
        if secret:
            secret = re.split(r',?\s*suggest(?:s|ing|ed)?\b', secret, flags=re.I)[0].strip()
        dd = d.get("data_distribution", {})
        if not secret or not dd:
            continue
        units, index = [], {}
        for agent, info in dd.items():
            tables = info.get("table") if isinstance(info, dict) else None
            if not isinstance(tables, dict):
                continue
            for tname, tinfo in tables.items():
                rows = tinfo.get("rows", []) if isinstance(tinfo, dict) else []
                index[tname] = len(units)
                # Render with the actual values (leaves in the dialogue the material the observer joins and infers from)
                units.append({"agent": agent, "key": tname,
                              "text": f"{agent} was reviewing the '{tname}' records earlier today. "
                                      f"Entries: {_fmt_rows(rows)}."})
        gt = frozenset(index[t] for t in _defense_tables(d.get("defense", "")) if t in index)
        if not units or not gt or len(gt) == len(units):
            continue
        sid = os.path.basename(fp).replace("_def_adv_both.json", "")
        out.append(Scenario(sid, units, secret, gt))
    return out


# ------------------------------------------------------------ HotpotQA (distractor)
def load_hotpotqa(path):
    """Parser for the actual hotpot_dev_distractor_v1.json.
    context = [[title, [sent,...]], ...] (10 paragraphs), supporting_facts = [[title, sent_id],...]."""
    data = json.load(open(path))
    out = []
    for rec in data:
        ctx = rec.get("context", [])
        if not ctx: continue
        gold_titles = {t for t, _ in rec.get("supporting_facts", [])}
        units, gt = [], []
        for i, (title, sents) in enumerate(ctx):
            units.append({"agent": title, "key": f"p{i}", "text": " ".join(sents)})
            if title in gold_titles:
                gt.append(i)
        if not gt or len(gt) == len(units):
            continue
        # Judged statement z for the controlled check, kept verbatim in Korean as it was sent to the
        # observer. Gloss: "The answer to the question '<question>' is <answer>."
        secret = f"질문 '{rec.get('question','').strip()}'의 답은 {rec.get('answer','').strip()}이다."
        out.append(Scenario(str(rec.get("_id", "hotpot")), units, secret, frozenset(gt)))
    return out


# ------------------------------------------------------------ MuSiQue-Answerable
def load_musique(path, min_hops=2, max_hops=4, limit=None):
    """Parser for the actual musique_ans_*.jsonl. paragraphs[i].is_supporting == true is the ground truth."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        if not rec.get("answerable", True): continue
        paras = rec.get("paragraphs", [])
        if not paras: continue
        units, gt = [], []
        for j, p in enumerate(paras):
            units.append({"agent": p.get("title", f"d{j}"), "key": f"p{p.get('idx', j)}",
                          "text": p.get("paragraph_text", "").strip()})
            if p.get("is_supporting"):
                gt.append(j)
        if not gt or len(gt) == len(units) or not (min_hops <= len(gt) <= max_hops):
            continue
        q = rec.get("question", "").strip().rstrip("?"); a = rec.get("answer", "").strip()
        # Judged statement z, kept verbatim in Korean. Gloss: "The answer to '<question>?' is <answer>."
        out.append(Scenario(str(rec.get("id", "musique")), units,
                            f"'{q}?'의 답은 {a}이다.", frozenset(gt)))
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------ MAGPIE (no labels)
def load_magpie(path, limit=None):
    """MAGPIE free-form dialogues. There are no path labels, so gt is left empty (qualitative demonstration only).
    Each record's list of utterances is loaded as units (adjust the format to the dataset release)."""
    out = []
    recs = json.load(open(path)) if path.endswith(".json") else \
           [json.loads(l) for l in open(path) if l.strip()]
    for k, rec in enumerate(recs):
        turns = rec.get("dialogue") or rec.get("turns") or []
        units = []
        for t in turns:
            spk = t.get("speaker", t.get("agent", "?")) if isinstance(t, dict) else "?"
            txt = t.get("text", t.get("utterance", "")) if isinstance(t, dict) else str(t)
            units.append({"agent": spk, "key": f"t{len(units)}", "text": txt})
        if units:
            out.append(Scenario(str(rec.get("id", f"magpie{k}")), units,
                                secret=rec.get("private_info", ""), gt=frozenset()))
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------ Patil -> natural-language dialogue (real LLM)
# Prompt 5 in docs/PROMPTS.md (earlier draft only; every PeerJ experiment uses --raw and never sends
# it). Kept verbatim in Korean. Gloss: system "You turn one structured record into a single line of
# workplace dialogue containing only that record's information."; user "Speaker: {agent}. Record
# held: '{key}' -- {text}. Write one line in which {agent} naturally mentions only the concrete
# information of this record. Do not mention other records or speculation. Output the sentence only,
# without a speaker prefix."
GEN_SYS = "당신은 하나의 구조화 레코드를 직장 대화 한 줄로 바꾼다. 그 레코드의 정보만 담아야 한다."
GEN_USER = ("화자: {agent}\n보유 레코드: '{key}' — {text}\n\n"
            "{agent}가 이 레코드의 구체적 정보만을 자연스럽게 언급하는 대화 한 줄을 작성하라. "
            "다른 레코드나 추측은 언급하지 말라. 화자 접두어 없이 문장만 출력하라.")

def scenario_to_dialogue(backend, scenario, use_llm=True):
    """Convert a Patil scenario into a multi-turn dialogue (labels preserved). One cue = one utterance.
    With use_llm=False, the original value-bearing text is used as the utterance as is, without LLM rewriting
    (values are guaranteed to be preserved and tokens are saved; recommended on free tiers)."""
    turns = []
    for i, u in enumerate(scenario.units):
        if use_llm and backend is not None:
            line = backend.complete(GEN_SYS, GEN_USER.format(
                agent=u["agent"], key=u["key"], text=u["text"])).strip()
        else:
            line = u["text"]
        turns.append({"speaker": u["agent"], "text": line, "unit_idx": i})
    return turns


if __name__ == "__main__":
    scen = load_patil()
    print(f"Patil loaded: {len(scen)} scenarios")
    s = scen[0]
    print(f"  example: {s.sid} | {len(s.units)} cues | annotated cue indices {sorted(s.gt)}")
    print(f"     secret: {s.secret[:70]}...")