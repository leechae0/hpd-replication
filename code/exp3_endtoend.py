# -*- coding: utf-8 -*-
"""
================================================================================
 Experiment 3 — end-to-end evaluation with real LLM observers (standalone; requires a real LLM)
================================================================================
 Pipeline: induction (A/B/C gate) → search (MR2) → validation (real spaCy-based MRs)
 Extra measurements: monotonicity-violation rate, two baselines (direct question / leave-one-out),
 coverage of real spurious cases.
 Data: Patil by default (converted to natural-language dialogue by a real LLM). MuSiQue/MAGPIE are
 path extensions / demonstrations.
 Usage:
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
    # Content words: Hangul syllables (가-힣) are included because some observers answered in Korean.
    zt = set(re.findall(r"[가-힣a-z0-9]+", z.lower()))
    for it in items:
        w = set(re.findall(r"[가-힣a-z0-9]+", str(it).lower()))
        if len(zt & w) >= max(2, len(zt) // 3):
            return True
    return False


# --------------- Search · monotonicity · validation (real LLM observer) ---------------
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
    # MR4: rename entities consistently in both the excerpt and the secret z before judging
    # (avoids false rejections caused by a name mismatch)
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
    """Harvest spurious cases in the real setting: (a) a YES on a content-masked dialogue means the
    conclusion comes from prior knowledge; (b) a YES for an unrelated secret z' is a misjudgment."""
    harvested = caught = 0
    # "(내용 은닉)" = "(content hidden)"; this placeholder is sent to the observer, so it is kept verbatim.
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
    """Aggregate over the progress file (all completed scenarios); runs split over several days are summed."""
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
    print("\n" + "=" * 68 + f"\nExperiment 3 results {label}({len(done)} scenarios aggregated)\n" + "=" * 68)
    print(f"[gate] induction A/B/C = {dict(abc)} → recall (A rate) = {abc['A']/n:.3f}")
    if jacs: print(f"[search]  mean path Jaccard {st.mean(jacs):.3f} · mean queries {st.mean(qtot):.0f} · paths found {len(jacs)}")
    if mono: print(f"[assumption]  mean monotonicity-violation rate {st.mean(mono):.4f}")
    if cert_tot:
        print(f"[validation]  all checks passed (genuine acceptance) {cert_pass}/{cert_tot} = {cert_pass/cert_tot:.3f}")
        print(f"[validation]  relaxed criterion (four of five) {cert_pass4}/{cert_tot} = {cert_pass4/cert_tot:.3f}")
        print(f"[per-check pass rate] " + " · ".join(
            f"{k} {mrpass[k]}/{cert_tot}({mrpass[k]/cert_tot:.2f})" for k in ("base", "single", "MR1", "MR3", "MR4")))
    if b_ask: print(f"[baseline] ask-the-model {st.mean(b_ask):.3f} · leave-one-out {st.mean(b_loo):.3f}"
                    + (f" (proposed {st.mean(jacs):.3f})" if jacs else ""))
    if harv: print(f"[real spurious] harvested {harv} · removed by validation {caught} ({caught/harv:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--mono", type=int, default=8, help="number of monotonicity samples (0 skips the probe and saves tokens)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the two baselines (saves tokens)")
    ap.add_argument("--no-harvest", action="store_true", help="skip the real-spurious coverage check (saves tokens)")
    ap.add_argument("--debug", action="store_true", help="print the benchmark secret, induction candidates and direct judgment per scenario (diagnostic)")
    ap.add_argument("--raw", action="store_true", help="render the dialogue with the values as is, without LLM rewriting (preserves values, saves tokens)")
    ap.add_argument("--start", type=int, default=0, help="start from this index")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between scenarios (eases rate limits)")
    ap.add_argument("--report-only", action="store_true", help="aggregate the progress file only, without running")
    ap.add_argument("--elicit-only", action="store_true", help="re-measure only the induction gate (A/B/C) of completed scenarios and update it (with the normalizing parser)")
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
        print(f"Induction re-measurement — {len(todo)} scenarios (backend={args.backend}, r={args.r}) · progress file {prog}")
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
                # Scenario was previously C, so no search was run → run search and validation now
                qc = QueryCounter(); S = find_path(obs, turns, z, qc); rec["q"] = qc.n
                if S is not None:
                    rec["S"] = sorted(S); rec["jac"] = jaccard(S, s.gt)
                    ok, cd = certify(obs, turns, S, z, detail=True)
                    rec["cert"] = bool(ok)
                    rec["cd"] = {kk: bool(cd[kk]) for kk in ("base", "single", "MR1", "MR3", "MR4")}
                    rec["cert4"] = sum(int(cd[kk]) for kk in ("base", "single", "MR1", "MR3", "MR4")) >= 4
                else:
                    rec["jac"] = None
            if (k + 1) % 10 == 0: print(f"  ...{k+1}/{len(todo)} (changed {changed})")
            if args.sleep: time.sleep(args.sleep)
        with open(prog, "w", encoding="utf-8") as f:
            for sid in [s.sid for s in scen if s.sid in done]:
                f.write(json.dumps(done[sid], ensure_ascii=False) + "\n")
        print(f"Induction re-measurement done · A/B/C changed for {changed} · progress file updated")
        report({k: v for k, v in done.items() if k in {s.sid for s in scen}}); return

    remain = [s for s in scen if s.sid not in done]
    print(f"Experiment 3 end-to-end — {len(scen)} scenarios (backend={args.backend}, r={args.r}) · "
          f"done {len(scen)-len(remain)} · this run {len(remain)} · progress file {prog}")
    print("  (if interrupted, re-run the same command: completed scenarios are skipped)")

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
                print(f"\n--- {s.sid} ---\n  benchmark secret: {z[:120]}\n  direct judgment: {'YES' if direct==1 else 'NO'}")
                print(f"  induction candidates ({len(el)}): " + " | ".join(str(x)[:60] for x in el[:6]))
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
                if args.debug: print(f"  validation: {cd}")
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
            if (k + 1) % 5 == 0: print(f"  ...{k+1}/{len(remain)} (completed so far {len(done)})")
            if args.sleep: time.sleep(args.sleep)
        except Exception as e:
            print(f"  [stopped] limit/error at {s.sid}: {str(e)[:140]}")
            print(f"  → aggregating the {len(done)} completed scenarios. After the limit resets, re-run the same command to resume.")
            break

    report({k: v for k, v in done.items() if k in {s.sid for s in scen}})


if __name__ == "__main__":
    main()
