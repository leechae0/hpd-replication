# -*- coding: utf-8 -*-
"""
exp3_stats.py — paired comparison of the three observers in Experiment 3 (Table VI of the paper).

All three observers were run on the same 116 Patil scenarios, so the comparison is paired:
  * binary outcomes (secret induction A, path returned, final 5/5 validation over 116):
    exact two-sided McNemar test on the discordant pairs, Holm-adjusted over the three
    pairwise comparisons of each measure;
  * Jaccard similarity to the benchmark-annotated cue set: paired bootstrap (10,000 resamples,
    seed 0) of the mean difference over scenarios in which both observers returned a path,
    plus a Wilcoxon signed-rank test;
  * overlap of the failure sets (final validation not passed).

Reads results/logs/e2e_progress_*_r5.jsonl (copied next to this script or given with --logs).
Writes results/exp3_stats.json and prints the table.
"""
import argparse, glob, itertools, json, math, os, random, statistics as st

OBS = [("GPT-4o-mini", "openai_gpt-4o-mini-2024-07-18"),
       ("Qwen3.6-27B", "openrouter_qwen_qwen3.6-27b"),
       ("Gemini 2.5 Flash-Lite", "openrouter_google_gemini-2.5-flash-lite")]


def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def wilcoxon_p(x, y):
    try:
        from scipy import stats
        return float(stats.wilcoxon(x, y).pvalue)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "logs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp3_stats.json"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--suffix", default="", help='log-name suffix, e.g. "_mr4fix" to use the MR4 re-judged logs')
    a = ap.parse_args()
    if a.suffix and a.out.endswith("exp3_stats.json"):
        a.out = a.out[:-len(".json")] + a.suffix + ".json"
    D = {}
    for name, key in OBS:
        f = glob.glob(os.path.join(a.logs, f"e2e_progress_{key}_r5{a.suffix}.jsonl"))
        assert f, key
        D[name] = {r["sid"]: r for r in map(json.loads, open(f[0], encoding="utf8"))}
    sids = sorted(set.intersection(*[set(v) for v in D.values()]))
    print("paired scenarios:", len(sids))
    ind = {m: {s: D[m][s]["abc"] == "A" for s in sids} for m in D}
    # a path counts as returned only in A/B cases, as in exp3_compare.py (Table V); C cases are excluded
    ret = {m: {s: ("S" in D[m][s] and D[m][s].get("jac") is not None and D[m][s]["abc"] != "C") for s in sids} for m in D}
    cert = {m: {s: bool(D[m][s].get("cert", False)) for s in sids} for m in D}
    out = {"n": len(sids), "rates": {}, "pairs": {}}
    for m in D:
        out["rates"][m] = {"induction": st.mean(ind[m].values()), "path_returned": st.mean(ret[m].values()), "final_5of5": st.mean(cert[m].values())}
    pairs = list(itertools.combinations(D, 2))
    rng = random.Random(0)
    for measure, table in (("induction", ind), ("path_returned", ret), ("final_5of5", cert)):
        ps = []
        for x, y in pairs:
            b01 = sum(1 for s in sids if table[x][s] and not table[y][s]); b10 = sum(1 for s in sids if not table[x][s] and table[y][s])
            p = mcnemar_exact(b01, b10); ps.append(p)
            out["pairs"].setdefault(f"{x} vs {y}", {})[measure] = {"only_first": b01, "only_second": b10, "p_exact": p}
        for (x, y), padj in zip(pairs, holm(ps)):
            out["pairs"][f"{x} vs {y}"][measure]["p_holm"] = padj
    # Jaccard: paired bootstrap on scenarios where both returned a path
    wps = []
    for x, y in pairs:
        both = [s for s in sids if ret[x][s] and ret[y][s]]
        jx = [D[x][s]["jac"] for s in both]; jy = [D[y][s]["jac"] for s in both]
        diffs = [u - v for u, v in zip(jx, jy)]
        boots = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(a.boot))
        wp = wilcoxon_p(jx, jy); wps.append(wp if wp is not None else 1.0)
        out["pairs"][f"{x} vs {y}"]["jaccard"] = {"n_both": len(both), "mean_diff": st.mean(diffs),
                                                 "ci95": [boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot) - 1]], "wilcoxon_p": wp}
    for (x, y), padj in zip(pairs, holm(wps)):
        out["pairs"][f"{x} vs {y}"]["jaccard"]["wilcoxon_p_holm"] = padj
    # failure-set overlap
    fail = {m: {s for s in sids if not cert[m][s]} for m in D}
    out["failure_overlap"] = {"failed_all": len(set.intersection(*fail.values())), "passed_all": sum(1 for s in sids if all(cert[m][s] for m in D)),
                              "pairwise_jaccard": {f"{x} vs {y}": len(fail[x] & fail[y]) / len(fail[x] | fail[y]) for x, y in pairs}}
    # print
    def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
    print(f"\n{'pair':<40}{'induction a/b, p(Holm)':<26}{'path a/b, p':<22}{'final 5/5 a/b, p':<22}{'Jaccard diff [95% CI] (n)':<30}")
    for x, y in pairs:
        r = out["pairs"][f"{x} vs {y}"]
        i, pr, c, j = r["induction"], r["path_returned"], r["final_5of5"], r["jaccard"]
        print(f"{x + ' vs ' + y:<40}{str(i['only_first']) + '/' + str(i['only_second']) + ', ' + fp(i['p_holm']):<26}"
              f"{str(pr['only_first']) + '/' + str(pr['only_second']) + ', ' + fp(pr['p_holm']):<22}"
              f"{str(c['only_first']) + '/' + str(c['only_second']) + ', ' + fp(c['p_holm']):<22}"
              f"{j['mean_diff']:+.3f} [{j['ci95'][0]:+.3f}, {j['ci95'][1]:+.3f}] ({j['n_both']})")
    fo = out["failure_overlap"]
    print(f"\nfailed with all three observers: {fo['failed_all']}; passed with all three: {fo['passed_all']}; "
          f"pairwise Jaccard of failure sets: " + ", ".join(f"{k}: {v:.2f}" for k, v in fo["pairwise_jaccard"].items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("written:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
