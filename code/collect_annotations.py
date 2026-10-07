# -*- coding: utf-8 -*-
"""
collect_annotations.py — reads the filled annotator workbooks (from make_annotation_xlsx.py) and
writes the labels back into the audit CSVs, so that the existing `score` commands work unchanged.

  python3 collect_annotations.py task1 작업1_유도판정_A.xlsx [작업1_유도판정_B.xlsx]
  python3 collect_annotations.py task2 작업2_변환판정_A.xlsx [작업2_변환판정_B.xlsx]

  first file  -> column ann1, second file -> column ann2 ('final' is left as it is)

  Second annotator on the labelled sample only (workbooks made with make_annotation_xlsx.py --mode sample):
  python3 collect_annotations.py task1 작업1_유도판정_2차판정자_<이름>.xlsx --as ann2 --subset
  python3 collect_annotations.py task2 작업2_변환판정_2차판정자_<이름>.xlsx --as ann2 --subset
      writes the labels into column ann2 for the rows in the workbook only, then prints raw agreement and
      Cohen's kappa against the existing 'final' labels and writes <sheet>_disagreements.csv
  task1 -> ../results/induction/annotation_sheet.csv   then: python3 exp3_induction.py score
  task2 -> ../results/mr_fidelity/fidelity_sheet.csv   then: python3 exp3_mr_fidelity.py score

With two annotators, rows where they disagree are written to <sheet>_disagreements.csv; after the two
annotators agree on a label, put it in the 'final' column of the main CSV and re-run `score`.
The script refuses to write if a workbook has blank or invalid labels (it lists the row numbers).
"""
import argparse, csv, os, sys
from openpyxl import load_workbook

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TASKS = {
    "task1": dict(csv=os.path.join(RES, "induction", "annotation_sheet.csv"), sheet="작업1_판정",
                  key=("sid", "observer"), labels={"M", "X", "A"}),
    "task2": dict(csv=os.path.join(RES, "mr_fidelity", "fidelity_sheet.csv"), sheet="작업2_판정",
                  key=("sid", "mr"), labels={"Y", "N"}),
}


def read_workbook(path, t):
    wb = load_workbook(path, data_only=True)
    ws, ks = wb[t["sheet"]], wb["_key"]
    head = [str(c.value or "") for c in ws[1]]
    c_lab = next(i for i, h in enumerate(head) if h.startswith("판정"))
    c_memo = next(i for i, h in enumerate(head) if h.startswith("메모"))
    key = {row[0]: tuple(row[1:1 + len(t["key"])]) for row in ks.iter_rows(min_row=2, values_only=True) if row[0] is not None}
    out, bad = {}, []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        lab = str(row[c_lab] or "").strip().upper()
        if lab not in t["labels"]:
            bad.append((row[0], lab or "(빈칸)"))
        out[key[row[0]]] = (lab, str(row[c_memo] or "").strip(), row[0])
    if len(out) != len(key):
        sys.exit(f"{path}: {len(out)} rows in the sheet but {len(key)} in _key — rows were added or deleted")
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS)); ap.add_argument("files", nargs="+")
    ap.add_argument("--as", dest="col", choices=["ann1", "ann2"], default=None,
                    help="with one file: write it into this column (default ann1)")
    ap.add_argument("--subset", action="store_true",
                    help="the workbook covers only some rows (e.g. the labelled sample); other rows are left as they are")
    a = ap.parse_args()
    t = TASKS[a.task]
    if a.col or a.subset:
        return second_pass(a, t)
    if len(a.files) > 2:
        sys.exit("at most two annotators (ann1, ann2)")
    anns = []
    for f in a.files:
        lab, bad = read_workbook(f, t)
        if bad:
            print(f"{f}: {len(bad)} rows blank or invalid (No: label) — nothing written")
            print("  " + ", ".join(f"{n}: {l}" for n, l in bad[:60]) + (" ..." if len(bad) > 60 else ""))
            sys.exit(1)
        anns.append(lab)

    rows = list(csv.DictReader(open(t["csv"], encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    for r in rows:
        k = tuple(r[f] for f in t["key"])
        for j, lab in enumerate(anns, 1):
            if k not in lab:
                sys.exit(f"row {k} of {t['csv']} is missing from {a.files[j - 1]}")
            r[f"ann{j}"] = lab[k][0]
    with open(t["csv"], "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"written: {t['csv']} ({len(rows)} rows, {len(anns)} annotator(s))")
    for j, lab in enumerate(anns, 1):
        cnt = {}
        for v in lab.values():
            cnt[v[0]] = cnt.get(v[0], 0) + 1
        print(f"  ann{j} ({os.path.basename(a.files[j - 1])}): " + "  ".join(f"{k}={cnt.get(k, 0)}" for k in sorted(t["labels"])))

    if len(anns) == 2:
        dis = [(anns[0][k][2], k, anns[0][k], anns[1][k]) for k in anns[0] if anns[0][k][0] != anns[1][k][0]]
        n = len(anns[0])
        print(f"  raw agreement: {1 - len(dis) / n:.3f} ({n - len(dis)}/{n}); disagreements: {len(dis)}")
        out = os.path.splitext(t["csv"])[0] + "_disagreements.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["No"] + list(t["key"]) + ["ann1", "memo1", "ann2", "memo2", "final"])
            for no, k, x, y in sorted(dis, key=lambda d: d[0]):
                w.writerow([no, *k, x[0], x[1], y[0], y[1], ""])
        print(f"  disagreements -> {out}  (agree on each, then copy the label into 'final' of the main CSV)")


def kappa(x, y, cats):
    n = len(x); po = sum(p == q for p, q in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return po, ((po - pe) / (1 - pe) if pe < 1 else float("nan"))


def second_pass(a, t):
    if len(a.files) != 1:
        sys.exit("--as/--subset take exactly one workbook")
    col = a.col or "ann1"
    lab, bad = read_workbook(a.files[0], t)
    if bad:
        print(f"{a.files[0]}: {len(bad)} rows blank or invalid (No: label) — nothing written")
        print("  " + ", ".join(f"{n}: {l}" for n, l in bad[:60]))
        sys.exit(1)
    rows = list(csv.DictReader(open(t["csv"], encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    seen = 0
    for r in rows:
        k = tuple(r[f] for f in t["key"])
        if k in lab:
            r[col] = lab[k][0]; seen += 1
        elif not a.subset:
            sys.exit(f"row {k} of {t['csv']} is missing from {a.files[0]}")
    if seen != len(lab):
        sys.exit(f"{len(lab) - seen} workbook rows do not match the CSV — nothing written")
    with open(t["csv"], "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"written: {seen} labels into column {col} of {t['csv']}")
    ref = [r for r in rows if tuple(r[f] for f in t["key"]) in lab and r.get("final", "").strip()]
    if not ref:
        return
    x = [r["final"].strip().upper() for r in ref]; y = [r[col].strip().upper() for r in ref]
    cats = sorted(t["labels"])
    po, k = kappa(x, y, cats)
    print(f"  against 'final' on {len(ref)} rows: raw agreement {po:.3f}, Cohen's kappa {k:.3f}")
    if a.task == "task1":
        po2, k2 = kappa(["M" if v == "M" else "N" for v in x], ["M" if v == "M" else "N" for v in y], ["M", "N"])
        print(f"  M versus not-M (the distinction the stratified estimate uses): agreement {po2:.3f}, kappa {k2:.3f}")
    out = os.path.splitext(t["csv"])[0] + f"_{col}_disagreements.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["No"] + list(t["key"]) + ["final", col, f"memo_{col}"])
        for r in ref:
            kk = tuple(r[f] for f in t["key"])
            if r["final"].strip().upper() != r[col].strip().upper():
                w.writerow([lab[kk][2], *kk, r["final"], r[col], lab[kk][1]])
    print(f"  disagreements -> {out}")


if __name__ == "__main__":
    main()
