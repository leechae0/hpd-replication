# -*- coding: utf-8 -*-
"""
================================================================================
 실험 3 — 실제 LLM 관찰자 종단 평가 (독립 실행, 실 LLM 필요)
================================================================================
 파이프라인: 유도(A/B/C 게이트) → 탐색(MR2) → 인증(실 spaCy MR)
 추가 측정: 단조성 위반율, 베이스라인 2종(직접질문/leave-one-out), 실가짜 coverage.
 데이터: Patil(실 LLM 으로 자연어 대화 변환) 기본. MuSiQue/MAGPIE 는 경로 확장/시연.
 실행:
   GROQ_API_KEY=..   python3 exp3_endtoend.py --backend groq   --limit 20
   OPENAI_API_KEY=.. python3 exp3_endtoend.py --backend openai --model gpt-4o-2024-08-06
================================================================================
"""
import argparse, os, json, random, statistics as st, re, time
from collections import Counter
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import backward_search, jaccard, QueryCounter
import mr_nlp as MR


def render(turns, idx=None):
    sel = turns if idx is None else [turns[i] for i in sorted(idx)]
    return "\n".join(f'{t["speaker"]}: {t["text"]}' for t in sel)

def render_indexed(turns):
    return "\n".join(f'[{i}] {t["speaker"]}: {t["text"]}' for i, t in enumerate(turns))

def mentions(items, z):
    zt = set(re.findall(r"[가-힣a-z0-9]+", z.lower()))
    for it in items:
        w = set(re.findall(r"[가-힣a-z0-9]+", str(it).lower()))
        if len(zt & w) >= max(2, len(zt) // 3):
            return True
    return False


# --------------- 탐색 · 단조성 · 인증 (실 LLM 관찰자) ---------------
def find_path(obs, turns, z, qc):
    V = lambda S: obs.judge_excerpt(render(turns, S), z)
    return backward_search(V, list(range(len(turns))), qc)

def monotonicity(obs, turns, z, samples=8, rng=None):
    rng = rng or random.Random(0); n = len(turns); viol = tested = 0
    for _ in range(samples):
        k = rng.randint(1, n - 1); S = set(rng.sample(range(n), k))
        rest = [i for i in range(n) if i not in S]
        if not rest: continue
        i = rng.choice(rest)
        if obs.judge_excerpt(render(turns, S), z) == 1 and \
           obs.judge_excerpt(render(turns, S | {i}), z) == 0:
            viol += 1
        tested += 1
    return viol / tested if tested else 0.0

def certify(obs, turns, S, z, detail=False):
    ex = render(turns, S)
    base = obs.judge_excerpt(ex, z) == 1
    single = all(obs.judge_excerpt(render(turns, [i]), z) == 0 for i in S) if len(S) >= 2 else False
    r1 = obs.judge_excerpt(MR.mr1_reframe(ex), z) == 1
    noised, _ = MR.mr3_inject_noise([turns[i] for i in sorted(S)], 2)
    r3 = obs.judge_excerpt(render(noised), z) == 1
    # MR4: 개체명을 발췌와 비밀 z에 '일관되게' 치환한 뒤 판정(이름 불일치로 인한 오탈락 방지)
    ex4, mp4, _ = MR.mr4_rename(ex)
    z4 = z
    for old, new in mp4.items():
        z4 = re.sub(rf"\b{re.escape(old)}\b", new, z4)
    r4 = obs.judge_excerpt(ex4, z4) == 1
    ok = base and single and r1 and r3 and r4 and len(S) >= 2
    if detail:
        return ok, {"base": base, "single": single, "MR1": r1, "MR3": r3, "MR4": r4, "|S|": len(S)}
    return ok

def harvest_coverage(obs, turns, z, wrong_z):
    """실환경 스퓨리어스 수확: (a) 내용 은닉 대화에서 결론=사전지식, (b) 엉뚱한 z'에 YES=오판."""
    harvested = caught = 0
    masked = [{"speaker": t["speaker"], "text": "(내용 은닉)"} for t in turns]
    if obs.judge_excerpt(render(masked), z) == 1:
        harvested += 1
        if not certify(obs, masked, set(range(len(masked))), z): caught += 1
    if obs.judge_excerpt(render(turns), wrong_z) == 1:
        harvested += 1
        if not certify(obs, turns, set(range(len(turns))), wrong_z): caught += 1
    return harvested, caught


def _prog_path(args):
    tag = (args.model or {"groq":"llama-3.3-70b-versatile","openai":"gpt-4o-2024-08-06",
                          "anthropic":"claude-3-5-sonnet-20241022","gemini":"gemini-2.5-flash",
                          "cerebras":"qwen-3.8-27b","openrouter":"qwen_qwen3.6-27b"}[args.backend])
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag)
    return f"e2e_progress_{args.backend}_{tag}_r{args.r}.jsonl"


def _load_progress(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line); done[rec["sid"]] = rec
    return done


def _save(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def report(done, label=""):
    """진행 파일(완료분 전체)에서 집계. 여러 날에 나눠 돌려도 합산된다."""
    abc = Counter(); jacs = []; qtot = []; mono = []
    cert_pass = cert_tot = cert_pass4 = 0; mrpass = Counter(); b_ask = []; b_loo = []
    harv = caught = 0
    for rec in done.values():
        abc[rec["abc"]] += 1
        if rec["abc"] == "C": continue
        if rec.get("jac") is not None:
            jacs.append(rec["jac"]); qtot.append(rec["q"])
            cert_tot += 1; cert_pass += int(rec["cert"]); cert_pass4 += int(rec["cert4"])
            for kk in ("base", "single", "MR1", "MR3", "MR4"): mrpass[kk] += int(rec["cd"][kk])
        if rec.get("mono") is not None: mono.append(rec["mono"])
        if rec.get("b_ask") is not None: b_ask.append(rec["b_ask"]); b_loo.append(rec["b_loo"])
        harv += rec.get("harv", 0); caught += rec.get("caught", 0)
    n = max(len(done), 1)
    print("\n" + "=" * 68 + f"\n실험 3 결과 {label}(집계 대상 {len(done)}개)\n" + "=" * 68)
    print(f"[게이트] 유도 A/B/C = {dict(abc)} → 재현율(A율) = {abc['A']/n:.3f}")
    if jacs: print(f"[탐색]  경로 자카드 평균 {st.mean(jacs):.3f} · 질의 평균 {st.mean(qtot):.0f} · 경로 확보 {len(jacs)}건")
    if mono: print(f"[가정]  단조성 위반율 평균 {st.mean(mono):.4f}")
    if cert_tot:
        print(f"[검증]  전체 통과(진성 수용률) {cert_pass}/{cert_tot} = {cert_pass/cert_tot:.3f}")
        print(f"[검증]  완화 기준(5중 4) {cert_pass4}/{cert_tot} = {cert_pass4/cert_tot:.3f}")
        print(f"[관계별 불변율] " + " · ".join(
            f"{k} {mrpass[k]}/{cert_tot}({mrpass[k]/cert_tot:.2f})" for k in ("base", "single", "MR1", "MR3", "MR4")))
    if b_ask: print(f"[베이스라인] 직접질문 {st.mean(b_ask):.3f} · leave-one-out {st.mean(b_loo):.3f}"
                    + (f" (제안 {st.mean(jacs):.3f})" if jacs else ""))
    if harv: print(f"[실가짜] 수확 {harv} · 검증이 걸러냄 {caught} ({caught/harv:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--mono", type=int, default=8, help="단조성 표본 수(0이면 생략, 토큰 절약)")
    ap.add_argument("--no-baseline", action="store_true", help="베이스라인 2종 생략(토큰 절약)")
    ap.add_argument("--no-harvest", action="store_true", help="실가짜 coverage 생략(토큰 절약)")
    ap.add_argument("--debug", action="store_true", help="시나리오별 정답 비밀·유도 후보·직접판정 출력(진단용)")
    ap.add_argument("--raw", action="store_true", help="대화를 LLM 재작성 없이 값 그대로 렌더(값 보존·토큰 절약)")
    ap.add_argument("--start", type=int, default=0, help="이 인덱스부터 시작")
    ap.add_argument("--sleep", type=float, default=0.0, help="시나리오 간 대기 초(rate limit 완화)")
    ap.add_argument("--report-only", action="store_true", help="실행 없이 진행 파일만 집계")
    ap.add_argument("--elicit-only", action="store_true", help="완료된 시나리오의 유도(A/B/C)만 다시 측정해 갱신(정규화 파서 적용)")
    args = ap.parse_args()

    prog = _prog_path(args)
    done = _load_progress(prog)
    scen = load_patil()
    scen = scen[args.start:] if not args.limit else scen[args.start:args.start + args.limit]
    if args.report_only:
        report({k: v for k, v in done.items() if k in {s.sid for s in scen}} or done, f"[{prog}] "); return

    backend = make_backend(args.backend, args.model)
    if args.elicit_only:
        obs = Observer(backend, r=args.r)
        todo = [s for s in scen if s.sid in done]
        print(f"유도 재측정 — {len(todo)}개 (backend={args.backend}, r={args.r}) · 진행 파일 {prog}")
        changed = 0
        for k, s in enumerate(todo):
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            z = s.secret; full = render(turns); rec = done[s.sid]
            el = obs.elicit(full); direct = obs.judge_excerpt(full, z)
            new_abc = "A" if mentions(el, z) else ("B" if direct == 1 else "C")
            if new_abc != rec.get("abc"):
                changed += 1
            rec["abc"] = new_abc
            if new_abc != "C" and "q" not in rec:
                # 이전엔 C라서 탐색을 안 했던 시나리오 → 탐색·검증 수행
                qc = QueryCounter(); S = find_path(obs, turns, z, qc); rec["q"] = qc.n
                if S is not None:
                    rec["S"] = sorted(S); rec["jac"] = jaccard(S, s.gt)
                    ok, cd = certify(obs, turns, S, z, detail=True)
                    rec["cert"] = bool(ok)
                    rec["cd"] = {kk: bool(cd[kk]) for kk in ("base", "single", "MR1", "MR3", "MR4")}
                    rec["cert4"] = sum(int(cd[kk]) for kk in ("base", "single", "MR1", "MR3", "MR4")) >= 4
                else:
                    rec["jac"] = None
            if (k + 1) % 10 == 0: print(f"  ...{k+1}/{len(todo)} (변경 {changed})")
            if args.sleep: time.sleep(args.sleep)
        with open(prog, "w", encoding="utf-8") as f:
            for sid in [s.sid for s in scen if s.sid in done]:
                f.write(json.dumps(done[sid], ensure_ascii=False) + "\n")
        print(f"유도 재측정 완료 · A/B/C 변경 {changed}건 · 진행 파일 갱신")
        report({k: v for k, v in done.items() if k in {s.sid for s in scen}}); return

    remain = [s for s in scen if s.sid not in done]
    print(f"실험 3 종단 — 대상 {len(scen)}개 (backend={args.backend}, r={args.r}) · "
          f"완료 {len(scen)-len(remain)} · 이번에 {len(remain)}개 · 진행 파일 {prog}")
    print("  (끊기면 같은 명령을 다시 실행하면 완료분은 건너뛰고 이어서 돕니다)")

    for k, s in enumerate(remain):
        try:
            obs = Observer(backend, r=args.r)
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            z = s.secret
            full = render(turns)
            rec = {"sid": s.sid}

            el = obs.elicit(full)
            direct = obs.judge_excerpt(full, z)
            if args.debug:
                print(f"\n--- {s.sid} ---\n  정답 비밀: {z[:120]}\n  직접 판정: {'YES' if direct==1 else 'NO'}")
                print(f"  유도 후보({len(el)}개): " + " | ".join(str(x)[:60] for x in el[:6]))
            if mentions(el, z): rec["abc"] = "A"
            elif direct == 1: rec["abc"] = "B"
            else:
                rec["abc"] = "C"; _save(prog, rec); done[s.sid] = rec
                if args.sleep: time.sleep(args.sleep)
                continue

            qc = QueryCounter()
            S = find_path(obs, turns, z, qc)
            rec["q"] = qc.n
            if S is not None:
                rec["S"] = sorted(S); rec["jac"] = jaccard(S, s.gt)
                ok, cd = certify(obs, turns, S, z, detail=True)
                rec["cert"] = bool(ok)
                rec["cd"] = {kk: bool(cd[kk]) for kk in ("base", "single", "MR1", "MR3", "MR4")}
                rec["cert4"] = sum(int(cd[kk]) for kk in ("base", "single", "MR1", "MR3", "MR4")) >= 4
                if args.debug: print(f"  검증: {cd}")
            else:
                rec["jac"] = None
            if args.mono > 0:
                rec["mono"] = monotonicity(obs, turns, z, samples=args.mono, rng=random.Random(hash(s.sid) & 0xffff))
            if not args.no_baseline:
                rep = obs.explain(render_indexed(turns), z); rec["b_ask"] = jaccard(rep, s.gt)
                loo = {i for i in range(len(turns))
                       if obs.judge_excerpt(render(turns, [x for x in range(len(turns)) if x != i]), z) == 0}
                rec["b_loo"] = jaccard(loo, s.gt)
            if not args.no_harvest:
                h, c = harvest_coverage(obs, turns, z, scen[(k + 1) % len(scen)].secret)
                rec["harv"] = h; rec["caught"] = c
            _save(prog, rec); done[s.sid] = rec
            if (k + 1) % 5 == 0: print(f"  ...{k+1}/{len(remain)} (누적 완료 {len(done)})")
            if args.sleep: time.sleep(args.sleep)
        except Exception as e:
            print(f"  [중단] {s.sid}에서 한도/오류: {str(e)[:140]}")
            print(f"  → 완료분 {len(done)}개로 집계합니다. 한도 리셋 후 같은 명령으로 이어서 돌리세요.")
            break

    report({k: v for k, v in done.items() if k in {s.sid for s in scen}})


if __name__ == "__main__":
    main()
