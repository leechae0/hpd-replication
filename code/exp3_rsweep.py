# -*- coding: utf-8 -*-
"""
================================================================================
 실험 3-4 (선택) — 검증 보정 스윕: r ∈ {1,3,5,9} × (전체 AND vs 5중 4) 비교
================================================================================
 목적: 실환경 진성 수용률 0.448이 '방법 결함'이 아니라 '보수적 관찰자 보정'의
 산물임을 실측으로 보이기 위한 실험. 경로 탐색은 r=5로 1회만 수행하고,
 같은 경로에 대해 검증 단계만 r을 바꿔 재판정한다(탐색 비용 재지불 없음).

 ★ 이어하기 지원: 시나리오별 결과를 rsweep_progress.jsonl 에 저장한다.
   끊겨도 같은 명령을 다시 실행하면 완료분은 건너뛰고 전체를 합산해 집계한다.
   처음부터 다시 하려면 rsweep_progress.jsonl 을 지우면 된다.

 실행:
   OPENAI_API_KEY=.. python3 exp3_rsweep.py --backend openai --model gpt-4o-mini \
       --limit 30 --raw --sleep 1
================================================================================
"""
import argparse, json, os, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import QueryCounter
from exp3_endtoend import render, find_path, certify

R_LIST = [1, 3, 5, 9]
KEYS = ("base", "single", "MR1", "MR3", "MR4")
PROG = "rsweep_progress.jsonl"


def load_progress():
    done = {}
    if os.path.exists(PROG):
        for line in open(PROG, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["sid"]] = rec
    return done


def save_record(rec):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def aggregate(done):
    tot = 0
    andp = {r_: 0 for r_ in R_LIST}
    p4 = {r_: 0 for r_ in R_LIST}
    rel = {r_: {k: 0 for k in KEYS} for r_ in R_LIST}
    for rec in done.values():
        if rec.get("no_path"):
            continue
        tot += 1
        for r_ in R_LIST:
            rr = rec["r"][str(r_)]
            andp[r_] += int(rr["and"])
            p4[r_] += int(rr["p4"])
            for k in KEYS:
                rel[r_][k] += int(rr["rel"][k])
    return tot, andp, p4, rel


def report(done):
    tot, andp, p4, rel = aggregate(done)
    n_skip = sum(1 for rec in done.values() if rec.get("no_path"))
    print("\n" + "=" * 64 + f"\n실험 3-4 결과 (경로 확보 {tot}개 · 경로 미확보 {n_skip}개)\n" + "=" * 64)
    print(f"{'r':>3} | {'전체 AND':>14} | {'5중 4':>14} | base·single·MR1·MR3·MR4")
    for r_ in R_LIST:
        if tot:
            rl = " ".join(f"{rel[r_][k]/tot:.2f}" for k in KEYS)
            print(f"{r_:>3} | {andp[r_]:>3}/{tot} = {andp[r_]/tot:.3f} | {p4[r_]:>3}/{tot} = {p4[r_]/tot:.3f} | {rl}")
    print("\n해석 지침: r 증가에 따라 AND 통과율이 오르면 0.448은 판정 요동(보정) 문제라는")
    print("주장이 실측으로 지지됨. 5중 4와 AND의 간격은 임계 완화 손잡이의 효과 크기.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model)
    scen = load_patil()[args.start:args.start + args.limit]
    done = load_progress()
    remain = [s for s in scen if s.sid not in done]
    print(f"실험 3-4 검증 보정 스윕 — 대상 {len(scen)}개 중 완료 {len(scen)-len(remain)}개, "
          f"이번에 돌릴 것 {len(remain)}개 (r={R_LIST})")

    for k, s in enumerate(remain):
        try:
            obs5 = Observer(backend, r=5)
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            z = s.secret
            qc = QueryCounter()
            S = find_path(obs5, turns, z, qc)      # 탐색은 r=5 고정 1회
            if S is None or len(S) < 2:
                rec = {"sid": s.sid, "no_path": True}
                save_record(rec); done[s.sid] = rec
                if args.sleep: time.sleep(args.sleep)
                continue
            rrec = {}
            for r_ in R_LIST:                       # 검증만 r 스윕
                obs = Observer(backend, r=r_)
                ok, cd = certify(obs, turns, S, z, detail=True)
                rrec[str(r_)] = {"and": int(ok),
                                 "p4": int(sum(int(cd[kk]) for kk in KEYS) >= 4),
                                 "rel": {kk: int(cd[kk]) for kk in KEYS}}
            rec = {"sid": s.sid, "S": sorted(S), "r": rrec}
            save_record(rec); done[s.sid] = rec
            if (k + 1) % 3 == 0:
                print(f"  ...{k+1}/{len(remain)} (누적 경로 확보 "
                      f"{sum(1 for x in done.values() if not x.get('no_path'))})")
            if args.sleep: time.sleep(args.sleep)
        except Exception as e:
            print(f"  [중단] {s.sid}에서 오류: {str(e)[:100]}")
            print(f"  → 지금까지의 완료분으로 집계합니다. 같은 명령으로 재실행하면 이어서 돕니다.")
            break

    report(done)


if __name__ == "__main__":
    main()
