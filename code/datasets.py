# -*- coding: utf-8 -*-
"""
datasets.py — 실제 데이터셋 로더 (mock 없음).
공통 스키마 Scenario 로 4개 데이터셋을 통일한다.

  Scenario.units  : 단서 목록. 각 단서 = {"agent","key","text"}
  Scenario.gt     : 정답 최소 단서 집합(단서 색인 frozenset). 없으면 비어 있음.
  Scenario.secret : 비밀 z (문자열)

레이블(gt)이 있는 데이터에서만 정량 채점이 가능하다:
  - Patil       : defense 필드 = 민감 조합 = 최소 지지집합           (gt 有)
  - HotpotQA    : supporting_facts 의 gold 문단                      (gt 有)
  - MuSiQue     : is_supporting == true 문단                          (gt 有)
  - MAGPIE      : 경로 레이블 없음 → 정성 시연 전용                    (gt 無)
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
# 데이터 경로 자동 탐색: 환경변수 PATIL_DIR > 프로젝트 동봉 data/ > 기본 경로.
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
    """레코드의 실제 값을 사람이 읽을 수 있게 렌더(관찰자가 조인할 값 보존)."""
    out = []
    for row in rows[:limit]:
        out.append("(" + ", ".join(f"{k}={v}" for k, v in row.items()) + ")")
    return "; ".join(out)

def load_patil(path=None):
    """Patil MultiAgentPrivacy 실 파일 로더. units = 에이전트별 테이블(단서)."""
    path = path or PATIL_DIR
    out = []
    for fp in sorted(glob.glob(os.path.join(path, "experiment_*_def_adv_both.json"))):
        d = json.load(open(fp))
        run = d.get("run_2_sensitive") or {}
        secret = run.get("final_inference_result")
        # 비밀을 '사실 결합' 부분까지로 한정(뒤의 추측 'suggesting…'은 데이터 너머 해석이라 제외).
        # 조합 누출의 실체는 단서들이 결정하는 사실 결합이며, 관찰자도 이 부분만 확정 가능.
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
                # 실제 값을 담아 렌더(관찰자가 조인·추론할 재료를 대화에 남긴다)
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
    """hotpot_dev_distractor_v1.json 실 파서.
    context = [[title, [sent,...]], ...] (문단 10개), supporting_facts = [[title, sent_id],...]."""
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
        secret = f"질문 '{rec.get('question','').strip()}'의 답은 {rec.get('answer','').strip()}이다."
        out.append(Scenario(str(rec.get("_id", "hotpot")), units, secret, frozenset(gt)))
    return out


# ------------------------------------------------------------ MuSiQue-Answerable
def load_musique(path, min_hops=2, max_hops=4, limit=None):
    """musique_ans_*.jsonl 실 파서. paragraphs[i].is_supporting == true 가 정답."""
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
        out.append(Scenario(str(rec.get("id", "musique")), units,
                            f"'{q}?'의 답은 {a}이다.", frozenset(gt)))
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------ MAGPIE (레이블 없음)
def load_magpie(path, limit=None):
    """MAGPIE 자유형 대화. 경로 레이블이 없으므로 gt는 비운다(정성 시연 전용).
    각 레코드의 발화 목록을 units 로 싣는다(형식은 데이터 판본에 맞게 조정)."""
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


# ------------------------------------------------------------ Patil → 자연어 대화 (실 LLM)
GEN_SYS = "당신은 하나의 구조화 레코드를 직장 대화 한 줄로 바꾼다. 그 레코드의 정보만 담아야 한다."
GEN_USER = ("화자: {agent}\n보유 레코드: '{key}' — {text}\n\n"
            "{agent}가 이 레코드의 구체적 정보만을 자연스럽게 언급하는 대화 한 줄을 작성하라. "
            "다른 레코드나 추측은 언급하지 말라. 화자 접두어 없이 문장만 출력하라.")

def scenario_to_dialogue(backend, scenario, use_llm=True):
    """Patil 시나리오를 다중턴 대화로 변환(라벨 보존). 단서 1개 = 발화 1개.
    use_llm=False면 LLM 재작성 없이 값이 담긴 원 텍스트를 그대로 발화로 사용
    (값 보존이 확실하고 토큰을 아낀다; 무료 계층 권장)."""
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
    print(f"Patil 로드: {len(scen)}개 시나리오")
    s = scen[0]
    print(f"  예: {s.sid} | 단서 {len(s.units)}개 | 정답 색인 {sorted(s.gt)}")
    print(f"     비밀: {s.secret[:70]}...")