# -*- coding: utf-8 -*-
"""
================================================================================
 negatives.py — builds negative (non-leaking) scenarios for dialogue-level detection (Experiment 3-2)
================================================================================
 Positives (Patil): dialogues in which a sensitive combination exists and forms a secret.
 Three kinds of negatives:
   (i)  distractor_only : only the cues of the sensitive combination (gt) are removed; distractors remain → no hidden secret
        (looks like a leaking dialogue: a hard negative that tests false positives from prior knowledge)
   (ii) shuffled        : cues from different scenarios are shuffled so that no consistent secret holds
   (iii) natural        : unlabelled free-form benign dialogues (MAGPIE), loaded by the detection script

 Each negative scenario is a Scenario(secret="", gt=frozenset()) and, like the positives, is turned into
 a natural-language dialogue by scenario_to_dialogue() before entering the pipeline.
================================================================================
"""
import random
from datasets import Scenario


def distractor_only(s):
    """Negative scenario with only the sensitive-combination (gt) cues removed and the distractor cues kept."""
    keep = [u for i, u in enumerate(s.units) if i not in s.gt]
    if len(keep) < 2:
        return None
    return Scenario(sid=f"neg_distract_{s.sid}", units=[dict(u) for u in keep],
                    secret="", gt=frozenset())


def shuffled(scen, rng, size=4):
    """Negative scenario with cues from different scenarios shuffled, so no consistent secret exists."""
    pool = [u for s in scen for u in s.units]
    if len(pool) < 2:
        return None
    pick = rng.sample(pool, min(size, len(pool)))
    return Scenario(sid="neg_shuffle", units=[dict(u) for u in pick],
                    secret="", gt=frozenset())


def build_negatives(scen, rng=None, n_shuffle=None):
    """Build the list of negative scenarios [(kind, Scenario), ...] from the positive list scen."""
    rng = rng or random.Random(0)
    negs = []
    for s in scen:
        d = distractor_only(s)
        if d:
            negs.append(("distractor", d))
    n_shuffle = n_shuffle if n_shuffle is not None else max(1, len(scen) // 2)
    for _ in range(n_shuffle):
        sh = shuffled(scen, rng, size=rng.randint(3, 5))
        if sh:
            negs.append(("shuffle", sh))
    return negs


if __name__ == "__main__":
    from datasets import load_patil
    pos = load_patil()
    negs = build_negatives(pos, rng=random.Random(0))
    from collections import Counter
    c = Counter(k for k, _ in negs)
    print(f"{len(pos)} positives → {len(negs)} negatives {dict(c)}")
    d = negs[0][1]
    print(f"  example (distractor): {len(d.units)} cues, secret='' gt={sorted(d.gt)}")
