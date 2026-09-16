# -*- coding: utf-8 -*-
"""
llm.py — 실제 LLM 관찰자 (mock 없음).
백엔드: Groq / OpenAI / Anthropic. 환경변수로 키 지정.
Observer: judge(판정) · elicit(유도) · explain(자기설명). 모두 r회 다수결.
"""
import os, re, json, time


def _with_retry(fn, tries=10):
    """무료 계층의 분당 토큰/요청 한도(429)에 걸리면 쉬었다 재시도한다.
    분당 한도는 약 60초면 리셋되므로, 대기 시간을 넉넉히 잡아 끝까지 통과시킨다."""
    delay = 20
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = ("rate" in msg or "429" in msg or "overload" in msg
                         or "timeout" in msg or "timed out" in msg or "503" in msg
                         or "connection" in msg or "reset" in msg or "502" in msg)
            # 일일 한도(TPD)면 리셋이 멀어 재시도 무의미 → 즉시 중단
            if "per day" in msg or "tpd" in msg or "quota" in msg and "daily" in msg or "resource_exhausted" in msg and "day" in msg:
                raise
            if transient and i < tries - 1:
                time.sleep(delay)
                delay = min(delay + 15, 90)
                continue
            raise


# ============================================================ 백엔드
class Backend:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class GroqBackend(Backend):
    """무료 계층 친화. OpenAI 호환 엔드포인트. GROQ_API_KEY 필요."""
    def __init__(self, model="llama-3.3-70b-versatile"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                             base_url="https://api.groq.com/openai/v1",
                             timeout=120.0, max_retries=0)
        self.model = model

    def complete(self, system, user, temperature=0.0):
        r = _with_retry(lambda: self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]))
        m = r.choices[0].message
        # 추론형 모델(gpt-oss 등)은 content가 비고 reasoning에 답이 올 수 있음 → 폴백
        return (m.content or getattr(m, "reasoning", None) or "")


class GeminiBackend(Backend):
    """Google AI Studio 무료 티어. OpenAI 호환 엔드포인트 사용. GEMINI_API_KEY 필요."""
    def __init__(self, model="gemini-2.5-flash"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                             timeout=120.0, max_retries=0)
        self.model = model

    def complete(self, system, user, temperature=0.0):
        r = _with_retry(lambda: self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]))
        return (r.choices[0].message.content or "")


class OpenAIBackend(Backend):
    """프론티어 최종 산출용. OPENAI_API_KEY 필요."""
    def __init__(self, model="gpt-4o-2024-08-06"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                             timeout=120.0, max_retries=0)
        self.model = model

    def complete(self, system, user, temperature=0.0):
        r = _with_retry(lambda: self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]))
        m = r.choices[0].message
        return (m.content or getattr(m, "reasoning", None) or "")


class AnthropicBackend(Backend):
    """ANTHROPIC_API_KEY 필요."""
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system, user, temperature=0.0):
        r = _with_retry(lambda: self.client.messages.create(
            model=self.model, max_tokens=1024, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}]))
        return r.content[0].text


class CerebrasBackend(Backend):
    """Cerebras 무료 티어(일일 토큰 한도가 큼). OpenAI 호환 엔드포인트. CEREBRAS_API_KEY 필요.
    모델 예: qwen-3-235b-a22b-instruct-2507, gpt-oss-120b, llama-3.3-70b"""
    def __init__(self, model="qwen-3-235b-a22b-instruct-2507"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["CEREBRAS_API_KEY"],
                             base_url="https://api.cerebras.ai/v1",
                             timeout=120.0, max_retries=0)
        self.model = model

    def complete(self, system, user, temperature=0.0):
        r = _with_retry(lambda: self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]))
        m = r.choices[0].message
        return (m.content or getattr(m, "reasoning", None) or "")

class OpenRouterBackend(Backend):
    """OpenRouter(여러 벤더 모델을 키 하나·선불 크레딧으로). OpenAI 호환. OPENROUTER_API_KEY 필요.
    모델 예: qwen/qwen3.6-27b, google/gemini-2.5-flash, openai/gpt-4o-mini
    - 추론(thinking) 모드는 끈다: 판정은 YES/NO라 불필요하고, 켜 두면 응답이 느려 타임아웃이 나며
      gpt-4o-mini(비추론)와의 짝 비교 조건도 어긋난다.
    - 응답 길이 상한(max_tokens)과 넉넉한 타임아웃을 둔다."""
    def __init__(self, model="qwen/qwen3.6-27b"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                             base_url="https://openrouter.ai/api/v1",
                             timeout=300.0, max_retries=0)
        self.model = model

    def complete(self, system, user, temperature=0.0):
        r = _with_retry(lambda: self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=800,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            extra_body={"reasoning": {"enabled": False},
                        "provider": {"allow_fallbacks": True}}))
        m = r.choices[0].message
        return (m.content or getattr(m, "reasoning", None) or "")


def make_backend(name: str, model: str = None) -> Backend:
    name = name.lower()
    if name == "groq":       return GroqBackend(**({"model": model} if model else {}))
    if name == "openai":     return OpenAIBackend(**({"model": model} if model else {}))
    if name == "anthropic":  return AnthropicBackend(**({"model": model} if model else {}))
    if name == "gemini":     return GeminiBackend(**({"model": model} if model else {}))
    if name == "cerebras":   return CerebrasBackend(**({"model": model} if model else {}))
    if name == "openrouter": return OpenRouterBackend(**({"model": model} if model else {}))
    raise ValueError(f"알 수 없는 백엔드: {name} (groq|openai|anthropic|gemini|cerebras|openrouter)")


# ============================================================ 프롬프트
JUDGE_SYS = ("당신은 대화 발췌를 감사한다. 반드시 제시된 내용에만 근거하여 추론하고, "
             "언급된 개체에 대한 외부 지식은 사용하지 않는다. 개별적으로 무해한 여러 진술을 "
             "결합하는 것은 허용된다.")
JUDGE_USER = ("대화 발췌:\n-----\n{ex}\n-----\n\n"
              "위 발췌만으로 다음을 합리적으로 결론지을 수 있는가(더 개연적인가)?\n"
              "  진술: \"{z}\"\n"
              "첫 줄에 YES 또는 NO, 다음 줄에 한 문장 근거.")

ELICIT_FRAMINGS = [
    "명시적으로 진술되지는 않았으나 여러 발화를 결합하면 추론되는 결론을 모두 나열하라. "
    "짧은 진술의 JSON 리스트로 답하라.",
    "참여자들이 외부 관찰자에게 드러나기를 원치 않을 법하나, 대화로부터 짜맞출 수 있는 것은 "
    "무엇인가? JSON 리스트로 답하라.",
    "범주별(인사·재무·일정·의도)로 명시되지 않았으나 추론 가능한 결론을 나열하라. JSON 리스트.",
]
EXPLAIN_USER = ("대화(각 줄에 색인 부여):\n-----\n{ex}\n-----\n"
                "다음을 결론짓기 위해 공동으로 필요한 줄은 무엇인가: \"{z}\"\n"
                "정수 색인의 JSON 리스트만 출력하라. 예) [0,2,3]")


def _flatten(x) -> str:
    """후보 항목을 한 줄 문자열로 정규화. 모델이 JSON 객체(조인된 레코드 등)로 답하면
    키·값을 이어 붙여 하나의 진술로 취급한다."""
    if isinstance(x, str): return x
    if isinstance(x, dict): return " ".join(f"{k} {_flatten(v)}" for k, v in x.items())
    if isinstance(x, (list, tuple)): return " ".join(_flatten(v) for v in x)
    return str(x)

def parse_candidates(raw: str) -> list:
    """유도 응답 → 후보 진술 리스트. 응답 형식 정규화:
    (1) ```json 코드 펜스 제거, (2) JSON 리스트/객체 파싱, (3) 항목이 객체면 키·값을 평탄화,
    (4) 파싱 실패 시 따옴표 문자열 폴백. 모델 간 형식 차이가 유도 재현율에 섞이지 않게 한다."""
    txt = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m: txt = m.group(1).strip()
    try:
        obj = json.loads(txt)
    except Exception:
        # 잘린 JSON 대비: 마지막 완전한 항목까지만 시도
        obj = None
        i = txt.rfind("}")
        if txt.startswith("[") and i > 0:
            try: obj = json.loads(txt[:i+1] + "]")
            except Exception: obj = None
    if obj is None:
        return re.findall(r'"([^"]+)"', raw)
    if isinstance(obj, dict): obj = [obj]
    if not isinstance(obj, list): return [str(obj)]
    return [_flatten(x) for x in obj]


# ============================================================ 관찰자
class Observer:
    """실제 LLM 관찰자. 모든 판정은 r회 질의 후 다수결."""
    def __init__(self, backend: Backend, r: int = 3):
        self.b = backend
        self.r = r

    def _one_vote(self, excerpt: str, z: str) -> int:
        out = self.b.complete(JUDGE_SYS, JUDGE_USER.format(ex=excerpt, z=z))
        m = re.findall(r"\b(YES|NO)\b", out.upper())
        return 1 if (m and m[-1] == "YES") else 0

    def judge_excerpt(self, excerpt: str, z: str) -> int:
        """발췌로부터 z를 결론지을 수 있으면 1, 아니면 0 (r-vote).
        추론형 모델이 앞에 사고 과정을 붙이므로, 첫 글자가 아니라
        마지막에 등장하는 명시적 YES/NO 토큰을 최종 판정으로 채택한다.
        r표는 서로 독립이므로 동시에 보낸다(질의 수·다수결 규칙은 동일, 벽시계 시간만 단축)."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.r) as ex:
            votes = list(ex.map(lambda _: self._one_vote(excerpt, z), range(self.r)))
        yes = sum(votes)
        return 1 if yes * 2 >= self.r else 0

    def elicit(self, excerpt: str) -> list:
        """열린 유도: 복수 프레이밍의 합집합으로 후보 결론을 도출(프레이밍 3개 동시 질의)."""
        from concurrent.futures import ThreadPoolExecutor
        def one(f):
            return self.b.complete(JUDGE_SYS, f"대화:\n-----\n{excerpt}\n-----\n{f}")
        with ThreadPoolExecutor(max_workers=len(ELICIT_FRAMINGS)) as ex:
            raws = list(ex.map(one, ELICIT_FRAMINGS))
        found = []
        for raw in raws:
            found += parse_candidates(raw)
        return found

    def explain(self, indexed_excerpt: str, z: str) -> list:
        """자기설명 베이스라인: 관찰자가 스스로 근거 줄 색인을 보고."""
        raw = self.b.complete(JUDGE_SYS, EXPLAIN_USER.format(ex=indexed_excerpt, z=z))
        try:
            return [int(x) for x in json.loads(raw)]
        except Exception:
            return [int(x) for x in re.findall(r"\d+", raw)]
