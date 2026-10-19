# -*- coding: utf-8 -*-
"""
================================================================================
 Experiment 3-3 — blocking (mitigation): validating the effectiveness of minimal blocking sets (standalone, real LLM)
================================================================================
 Response to a reviewer comment: addresses empirically the concern that Contribution 3
 (minimal blocking set) remained purely theoretical.
 Procedure (per scenario):
   1) Collect certified paths: search (find_path) -> certify (certify). Enumerate multiple
      paths (P) with one round of re-search that excludes the cues of the found path.
   2) Greedy hitting set: repeatedly select the cue that covers the most paths -> blocking set B.
   3) Mask the utterances in B as "(content withheld)".
   4) Re-measure after blocking: (a) direct judgment V(masked dialogue, z)  (b) whether open
      elicitation rediscovers z.
 Reported metrics:
   - mean blocking-set size |B| and its ratio to the dialogue, |B|/|U|
   - direct-judgment YES rate before blocking (baseline) vs. after blocking (residual leakage, judgment)
   - elicitation rediscovery rate after blocking (residual leakage, elicitation)
 Run (recommended):
   OPENAI_API_KEY=.. python3 exp3_block.py --backend openai --model gpt-4o-mini \
       --limit 30 --r 5 --raw --sleep 1
================================================================================
"""
import argparse, random, re, time
from llm import make_backend, Observer
from datasets import load_patil, scenario_to_dialogue
from search import QueryCounter
from exp3_endtoend import render, find_path, certify


def _secret_match(cands, secret):
    st = {w for w in re.findall(r"[a-z0-9]{3,}", secret.lower())}
    for c in cands:
        cw = {w for w in re.findall(r"[a-z0-9]{3,}", str(c).lower())}
        if len(st & cw) >= 2:
            return True
    return False


def enumerate_paths(obs, turns, z, max_paths=2):
    """Enumerate paths that pass certification: base search + one round of re-search (excluding the found path)."""
    paths = []
    qc = QueryCounter()
    S1 = find_path(obs, turns, z, qc)
    if S1 is not None and len(S1) >= 2 and certify(obs, turns, S1, z):
        paths.append(set(S1))
        # Exclusion re-search: try a second path among the cues remaining after removing those of S1
        keep = [i for i in range(len(turns)) if i not in S1]
        if len(keep) >= 2:
            sub = [turns[i] for i in keep]
            qc2 = QueryCounter()
            S2r = find_path(obs, sub, z, qc2)
            if S2r is not None and len(S2r) >= 2:
                S2 = {keep[i] for i in S2r}  # map back to original indices
                sub_sel = [turns[i] for i in sorted(S2)]
                if certify(obs, turns, S2, z):
                    paths.append(S2)
    return paths[:max_paths]


def greedy_hitting_set(paths):
    """Greedy approximation: at each step, select the cue that covers the most not-yet-hit paths."""
    B, remaining = set(), [set(p) for p in paths]
    while remaining:
        # number of paths covered by each candidate cue
        from collections import Counter
        cnt = Counter()
        for p in remaining:
            for u in p:
                cnt[u] += 1
        best = max(cnt, key=lambda u: cnt[u])
        B.add(best)
        remaining = [p for p in remaining if best not in p]
    return B


def mask(turns, B):
    return [{"speaker": t["speaker"],
             "text": "(content withheld)" if i in B else t["text"]}
            for i, t in enumerate(turns)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["groq", "openai", "anthropic", "gemini", "cerebras", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--raw", action="store_true", help="render values as is (same as Experiment 3-1; recommended)")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model)
    scen = load_patil()
    scen = scen[args.start:args.start + args.limit] if args.limit else scen[args.start:]
    print(f"Experiment 3-3 blocking — {len(scen)} scenarios (backend={args.backend}, r={args.r})")

    n_cert = 0            # number of scenarios with at least one certified path
    sizes, ratios = [], []
    pre_yes = post_yes = post_elicit = 0
    multi = 0             # scenarios with two or more enumerated paths

    done = 0
    for k, s in enumerate(scen):
        try:
            obs = Observer(backend, r=args.r)
            turns = scenario_to_dialogue(backend, s, use_llm=not args.raw)
            z = s.secret

            paths = enumerate_paths(obs, turns, z)
            done += 1
            if not paths:
                if (k + 1) % 5 == 0: print(f"  ...{k+1}/{len(scen)}")
                if args.sleep: time.sleep(args.sleep)
                continue
            n_cert += 1
            if len(paths) >= 2: multi += 1

            B = greedy_hitting_set(paths)
            sizes.append(len(B)); ratios.append(len(B) / len(turns))

            # Baseline: direct judgment before blocking (usually YES, since a certified path exists)
            pre_yes += int(obs.judge_excerpt(render(turns), z) == 1)
            # After blocking: direct judgment + rediscovery via open elicitation
            blocked = mask(turns, B)
            post_yes += int(obs.judge_excerpt(render(blocked), z) == 1)
            cands = obs.elicit(render(blocked))
            post_elicit += int(_secret_match(cands, z))

            if (k + 1) % 5 == 0: print(f"  ...{k+1}/{len(scen)}")
            if args.sleep: time.sleep(args.sleep)
        except Exception as e:
            print(f"  [stopped] limit/error at item {k+1}: {str(e)[:120]}")
            print(f"  → aggregating the {done} completed scenarios.")
            break

    print("\n" + "=" * 62 + "\nExperiment 3-3 blocking (mitigation) results\n" + "=" * 62)
    print(f"scenarios with a validated path: {n_cert}/{done} (two or more paths: {multi})")
    if sizes:
        import statistics as st
        print(f"blocking set size |B| mean {st.mean(sizes):.2f} · fraction of dialogue {st.mean(ratios):.1%}")
        print(f"direct-judgment YES rate  before blocking {pre_yes}/{n_cert} → after blocking {post_yes}/{n_cert}"
              f"  (residual leakage, judgment {post_yes/n_cert:.3f})")
        print(f"re-induction rate after blocking {post_elicit}/{n_cert} = {post_elicit/n_cert:.3f}  (residual leakage, induction)")
        print("Reading: the lower both residual rates after blocking, the more effective the targeted mitigation. What remains")
        print("      measures re-inference from prior knowledge (see the limitations discussion).")


if __name__ == "__main__":
    main()
