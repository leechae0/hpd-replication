# -*- coding: utf-8 -*-
"""
llm.py — real LLM observers (no mocks).
Backends: Groq / OpenAI / Anthropic. API keys are given via environment variables.
Observer: judge (verdict) · elicit (elicitation) · explain (self-explanation). All use an r-vote majority.
"""
import os, re, json, time


def _with_retry(fn, tries=10):
    """On the free tier, wait and retry when the per-minute token/request limit (429) is hit.
    The per-minute limit resets in about 60 s, so the waits are generous enough to get every call through."""
    delay = 20
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = ("rate" in msg or "429" in msg or "overload" in msg
                         or "timeout" in msg or "timed out" in msg or "503" in msg
                         or "connection" in msg or "reset" in msg or "502" in msg)
            # Daily limit (TPD): the reset is far away, so retrying is pointless -> abort immediately
            if "per day" in msg or "tpd" in msg or "quota" in msg and "daily" in msg or "resource_exhausted" in msg and "day" in msg:
                raise
            if transient and i < tries - 1:
                time.sleep(delay)
                delay = min(delay + 15, 90)
                continue
            raise


# ============================================================ Backends
class Backend:
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class GroqBackend(Backend):
    """Free-tier friendly. OpenAI-compatible endpoint. Requires GROQ_API_KEY."""
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
        # Reasoning models (e.g., gpt-oss) may leave content empty and put the answer in reasoning -> fall back
        return (m.content or getattr(m, "reasoning", None) or "")


class GeminiBackend(Backend):
    """Google AI Studio free tier. Uses the OpenAI-compatible endpoint. Requires GEMINI_API_KEY."""
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
    """For the final frontier-model runs. Requires OPENAI_API_KEY."""
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
    """Requires ANTHROPIC_API_KEY."""
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
    """Cerebras free tier (large daily token limit). OpenAI-compatible endpoint. Requires CEREBRAS_API_KEY.
    Example models: qwen-3-235b-a22b-instruct-2507, gpt-oss-120b, llama-3.3-70b"""
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
    """OpenRouter (models from many vendors with one key and prepaid credit). OpenAI-compatible. Requires OPENROUTER_API_KEY.
    Example models: qwen/qwen3.6-27b, google/gemini-2.5-flash, openai/gpt-4o-mini
    - Reasoning (thinking) mode is turned off: verdicts are YES/NO, so it is unnecessary; if left on, responses are slow and time out,
      and the matched-comparison condition with gpt-4o-mini (non-reasoning) no longer holds.
    - A response-length cap (max_tokens) and a generous timeout are set."""
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
    raise ValueError(f"Unknown backend: {name} (groq|openai|anthropic|gemini|cerebras|openrouter)")


# ============================================================ Prompts
# The observer prompts below are experimental inputs and are kept verbatim in Korean, exactly as
# they were sent to the observers. English glosses are given next to each prompt here and in
# docs/PROMPTS.md (prompt numbers refer to that file); the glosses were never sent to a model.

# Prompt 1 (system prompt, all calls). Gloss: "You audit dialogue excerpts. Reason only from the
# content shown; do not use outside knowledge about the entities mentioned. Combining several
# individually harmless statements is allowed."
JUDGE_SYS = ("당신은 대화 발췌를 감사한다. 반드시 제시된 내용에만 근거하여 추론하고, "
             "언급된 개체에 대한 외부 지식은 사용하지 않는다. 개별적으로 무해한 여러 진술을 "
             "결합하는 것은 허용된다.")
# Prompt 2 (binary judgment V_z(S)). Gloss: "Dialogue excerpt: ... Can the following be reasonably
# concluded (made more likely) from the excerpt alone? Statement: "{z}". First line YES or NO,
# second line a one-sentence reason."
JUDGE_USER = ("대화 발췌:\n-----\n{ex}\n-----\n\n"
              "위 발췌만으로 다음을 합리적으로 결론지을 수 있는가(더 개연적인가)?\n"
              "  진술: \"{z}\"\n"
              "첫 줄에 YES 또는 NO, 다음 줄에 한 문장 근거.")

# Prompt 3 (secret induction), three framings sent in parallel. Glosses:
#  (a) "List every conclusion that is not stated explicitly but can be inferred by combining several
#      utterances. Answer as a JSON list of short statements."
#  (b) "What could be pieced together from the dialogue that the participants would probably not want
#      an outside observer to learn? Answer as a JSON list."
#  (c) "By category (personnel, finance, schedule, intent), list conclusions that are not stated but
#      can be inferred. JSON list."
ELICIT_FRAMINGS = [
    "명시적으로 진술되지는 않았으나 여러 발화를 결합하면 추론되는 결론을 모두 나열하라. "
    "짧은 진술의 JSON 리스트로 답하라.",
    "참여자들이 외부 관찰자에게 드러나기를 원치 않을 법하나, 대화로부터 짜맞출 수 있는 것은 "
    "무엇인가? JSON 리스트로 답하라.",
    "범주별(인사·재무·일정·의도)로 명시되지 않았으나 추론 가능한 결론을 나열하라. JSON 리스트.",
]
# Prompt 4 (ask-the-model baseline; earlier draft only, not reported in the PeerJ manuscript).
# Gloss: "Dialogue (each line indexed): ... Which lines are jointly needed to conclude "{z}"? Output
# only a JSON list of integer indices, e.g. [0,2,3]."
EXPLAIN_USER = ("대화(각 줄에 색인 부여):\n-----\n{ex}\n-----\n"
                "다음을 결론짓기 위해 공동으로 필요한 줄은 무엇인가: \"{z}\"\n"
                "정수 색인의 JSON 리스트만 출력하라. 예) [0,2,3]")


def _flatten(x) -> str:
    """Normalize a candidate item to a one-line string. If the model answers with a JSON object (e.g., a joined record),
    its keys and values are concatenated and treated as a single statement."""
    if isinstance(x, str): return x
    if isinstance(x, dict): return " ".join(f"{k} {_flatten(v)}" for k, v in x.items())
    if isinstance(x, (list, tuple)): return " ".join(_flatten(v) for v in x)
    return str(x)

def parse_candidates(raw: str) -> list:
    """Elicitation response -> list of candidate statements. Normalizes the response format:
    (1) strip ```json code fences, (2) parse a JSON list/object, (3) flatten keys/values when an item is an object,
    (4) fall back to quoted strings if parsing fails. Keeps format differences between models out of elicitation recall."""
    txt = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m: txt = m.group(1).strip()
    try:
        obj = json.loads(txt)
    except Exception:
        # Guard against truncated JSON: try only up to the last complete item
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


# ============================================================ Observer
class Observer:
    """Real LLM observer. Every verdict is a majority vote over r queries."""
    def __init__(self, backend: Backend, r: int = 3):
        self.b = backend
        self.r = r

    def _one_vote(self, excerpt: str, z: str) -> int:
        out = self.b.complete(JUDGE_SYS, JUDGE_USER.format(ex=excerpt, z=z))
        m = re.findall(r"\b(YES|NO)\b", out.upper())
        return 1 if (m and m[-1] == "YES") else 0

    def judge_excerpt(self, excerpt: str, z: str) -> int:
        """1 if z can be concluded from the excerpt, else 0 (r-vote).
        Reasoning models prepend their thought process, so the last explicit YES/NO token that appears,
        not the first character, is taken as the final verdict.
        The r votes are independent, so they are sent concurrently (same query count and majority rule; only wall-clock time drops)."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.r) as ex:
            votes = list(ex.map(lambda _: self._one_vote(excerpt, z), range(self.r)))
        yes = sum(votes)
        return 1 if yes * 2 >= self.r else 0

    def elicit(self, excerpt: str) -> list:
        """Open elicitation: derive candidate conclusions as the union over several framings (the 3 framings are queried concurrently)."""
        from concurrent.futures import ThreadPoolExecutor
        def one(f):
            # "대화:" = "Dialogue:" (part of prompt 3, kept verbatim)
            return self.b.complete(JUDGE_SYS, f"대화:\n-----\n{excerpt}\n-----\n{f}")
        with ThreadPoolExecutor(max_workers=len(ELICIT_FRAMINGS)) as ex:
            raws = list(ex.map(one, ELICIT_FRAMINGS))
        found = []
        for raw in raws:
            found += parse_candidates(raw)
        return found

    def explain(self, indexed_excerpt: str, z: str) -> list:
        """Self-explanation baseline: the observer itself reports the indices of the supporting lines."""
        raw = self.b.complete(JUDGE_SYS, EXPLAIN_USER.format(ex=indexed_excerpt, z=z))
        try:
            return [int(x) for x in json.loads(raw)]
        except Exception:
            return [int(x) for x in re.findall(r"\d+", raw)]
