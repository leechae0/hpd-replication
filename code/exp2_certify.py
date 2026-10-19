# -*- coding: utf-8 -*-
"""
================================================================================
 Experiment 2 — validity of metamorphic certification (standalone)
================================================================================
 MR transformations are performed by real NLP code (mr_nlp: spaCy NER, coreference, retrieval), not a mock.
 As a controlled experiment, it uses observers with injected genuine/spurious judgment behavior.
   - Genuine observer: YES if all ground-truth cues are mentioned. Invariant to rephrasing, renaming, and irrelevant insertion.
   - Four spurious observers: each vulnerable to one failure type (surface wording, entity names,
     instability to irrelevant content, chance correlation).
 (Note: "injection" here is an experimental device for controlling observer behavior; the transformations themselves are real NLP code.)

 Measures:
   [0] implementation check : does each transformation behave as specified (deterministic checks)
   [1] soundness            : do genuine paths pass certification (≈ recall)
   [2] coverage             : are spurious paths rejected (≈ precision), over a pool of independent types
   [3] non-redundancy       : if one check is removed, does the corresponding type pass (ablation)
 Run: python3 exp2_certify.py   (requires spaCy en_core_web_sm)
================================================================================
"""
import re, random, statistics as st
from collections import Counter
from datasets import load_patil, scenario_to_dialogue  # scenario_to_dialogue is for real LLM conversion (optional)
import mr_nlp as MR


# --- Controlled observer: based on natural-language utterances. YES if all ground-truth table names are mentioned. ---
def _phrase(u):  # noun phrase identifying the cue (appears in the dialogue sentence)
    return u["key"]

class ControlledObserver:
    """Controlled observer with genuine (brittle=None) or spurious (brittle set) behavior."""
    def __init__(self, scenario, brittle=None):
        self.keys = {scenario.units[i]["key"] for i in scenario.gt}
        self.names = set(scenario.entities)
        self.brittle = brittle

    def judge_excerpt(self, excerpt, z=None):
        low = excerpt.lower()
        if not all(k.lower() in low for k in self.keys):
            return 0
        # Surface-wording dependence: breaks when MR1 rephrasing changes 'earlier today' (-> 'before this morning')
        # (matches the current utterance rendering in datasets.py: "... was reviewing the '<table>' records earlier today.")
        if self.brittle == "surface" and "earlier today" not in low:
            return 0
        # Entity-name dependence: breaks when MR4 renaming removes the original names
        if self.brittle == "entity" and not any(n.lower() in low for n in self.names):
            return 0
        # Unstable to irrelevant content: breaks when MR3 inserts an irrelevant utterance without 'records'
        if self.brittle == "noise" and any(
                ln.strip() and "records" not in ln for ln in low.splitlines()):
            return 0
        # cooccur (chance correlation): invariant to every transformation -> caught by no MR (an honestly reported gap)
        return 1


# Render the dialogue so that cue keys appear in the text (for the controlled experiment)
def render(units, idx=None):
    sel = units if idx is None else [units[i] for i in sorted(idx)]
    return "\n".join(f'{u.get("agent") or u.get("speaker") or "?"}: {u["text"]}' for u in sel)


def certify(observer, units, S):
    ex = render(units, S)
    base = observer.judge_excerpt(ex) == 1
    single = all(observer.judge_excerpt(render(units, [i])) == 0 for i in S) if len(S) >= 2 else False
    r1 = observer.judge_excerpt(MR.mr1_reframe(ex)) == 1
    noised, _ = MR.mr3_inject_noise([units[i] for i in sorted(S)], 2)
    r3 = observer.judge_excerpt(render(noised)) == 1
    ex4, _mp, _ = MR.mr4_rename(ex)
    r4 = observer.judge_excerpt(ex4) == 1
    return {"single": single, "MR1": r1, "MR3": r3, "MR4": r4,
            "certified": base and single and r1 and r3 and r4 and len(S) >= 2}

def certify_drop(observer, units, S, drop):
    r = certify(observer, units, S)
    keep = {k: (True if k == drop else r[k]) for k in ("single", "MR1", "MR3", "MR4")}
    return all(keep.values()) and observer.judge_excerpt(render(units, S)) == 1 and len(S) >= 2


# --- [Measure 0] Transformation implementation check (real NLP, deterministic) ---
def measure0(scen):
    agg = Counter(); tot = Counter()
    for s in scen:
        ex = render(s.units, s.gt)
        v1 = MR.verify_mr1(ex, MR.mr1_reframe(ex))
        tot["MR1_persons_kept"] += 1; agg["MR1_persons_kept"] += v1["persons_preserved"]
        r4, mp, _ = MR.mr4_rename(ex)
        v4 = MR.verify_mr4(ex, r4, mp)
        tot["MR4_no_old_name"] += 1; agg["MR4_no_old_name"] += v4["no_original_name_left"]
        noised, picked = MR.mr3_inject_noise(s.units, 2)
        v3 = MR.verify_mr3(s.units, picked)
        tot["MR3_added_unrelated"] += 1; agg["MR3_added_unrelated"] += v3["all_added_unrelated"]
    return {k: (agg[k], tot[k]) for k in tot}


if __name__ == "__main__":
    scen = load_patil()
    print(f"Experiment 2 (real spaCy MRs) — Patil, {len(scen)} scenarios\n")

    print("[Measure 0] implementation check (transformations as specified — deterministic)")
    for k, (ok, t) in measure0(scen).items():
        print(f"   {'OK ' if ok==t else 'CHK'} {k:<20} {ok}/{t}")

    ok = sum(certify(ControlledObserver(s), s.units, set(s.gt))["certified"] for s in scen)
    print(f"\n[Measure 1] soundness (genuine paths kept ≈ recall): {ok}/{len(scen)} = {ok/len(scen):.3f}")

    kinds = ["surface", "entity", "noise", "cooccur"]
    caught, tot = Counter(), Counter()
    for s in scen:
        for k in kinds:
            tot[k] += 1
            if not certify(ControlledObserver(s, brittle=k), s.units, set(s.gt))["certified"]:
                caught[k] += 1
    cov = sum(caught.values()) / sum(tot.values())
    print(f"\n[Measure 2] coverage (spurious paths caught ≈ precision): {cov:.1%}")
    for k in kinds:
        note = "  ← residual error (coincidental co-occurrence, reported as is)" if k == "cooccur" and caught[k] == 0 else ""
        print(f"   {k:<8} {caught[k]}/{tot[k]}{note}")

    print(f"\n[Measure 3] non-redundancy (paths of this type that pass when the check is removed):")
    for k, mr in {"surface": "MR1", "entity": "MR4", "noise": "MR3"}.items():
        leak = sum(1 for s in scen if certify_drop(ControlledObserver(s, brittle=k), s.units, set(s.gt), mr))
        print(f"   {mr} removed → {k} type passes: {leak}/{len(scen)}")
