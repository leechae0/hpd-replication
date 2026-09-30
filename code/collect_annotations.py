# -*- coding: utf-8 -*-
"""
collect_annotations.py — reads the filled annotator workbooks (from make_annotation_xlsx.py) and
writes the labels back into the audit CSVs, so that the existing `score` commands work unchanged.

  python3 collect_annotations.py task1 작업1_유도판정_A.xlsx [작업1_유도판정_B.xlsx]
  python3 collect_annotations.py task2 작업2_변환판정_A.xlsx [작업2_변환판정_B.xlsx]

  first file  -> column ann1, second file -> column ann2 ('final' is left as it is)
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
    a = ap.parse_args()
    t = TASKS[a.task]
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


if __name__ == "__main__":
    main()
