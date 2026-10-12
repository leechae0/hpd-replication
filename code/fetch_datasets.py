# -*- coding: utf-8 -*-
"""
================================================================================
 fetch_datasets.py (v2) — downloads HotpotQA and MuSiQue directly as HuggingFace parquet and
 saves them as original-format files that the loaders used by exp1_search.py read as is.
================================================================================
 Why parquet directly?
   - `datasets.load_dataset` breaks on Python 3.14 in its internal pickling (dill/tqdm), and
     hotpotqa/hotpot_qa is 'script-based', so recent versions of datasets refuse to load it.
   - HF automatically converts every dataset to parquet (refs/convert/parquet), so downloading
     only that parquet with huggingface_hub and reading it with pyarrow avoids both problems.
     (no heavy datasets runtime / trust-remote-code needed -> works on 3.14 too)

 Setup: pip install -U huggingface_hub pyarrow
 Run:   cd hpd && python3 fetch_datasets.py            # both
        python3 fetch_datasets.py --only hotpotqa
        python3 fetch_datasets.py --only musique --limit 500

 Then:  python3 exp1_search.py \
            --hotpotqa data/hotpot_dev_distractor_v1.json \
            --musique  data/musique_ans_v1.0_dev.jsonl
================================================================================
"""
import os, json, argparse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


def _read_parquet_rows(repo_id, revision, want):
    """Download every parquet file of repo_id whose path contains want (e.g. 'validation')
    and return the rows as a list of dicts. Uses huggingface_hub+pyarrow without the datasets library."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    parts = sorted(f for f in files
                   if f.endswith(".parquet") and want in f.lower())
    if not parts:
        raise FileNotFoundError(
            f"{repo_id}@{revision} 에서 '{want}' parquet 을 못 찾음. "
            f"전체 parquet: {[f for f in files if f.endswith('.parquet')][:8]} …")
    rows = []
    for f in parts:
        local = hf_hub_download(repo_id, f, repo_type="dataset", revision=revision)
        rows.extend(pq.read_table(local).to_pylist())
    return rows


# ------------------------------------------------------------ HotpotQA
def fetch_hotpotqa(limit=None):
    """hotpotqa/hotpot_qa (distractor, validation) parquet → saved in the original JSON schema.
    parquet columns: context={title:[...], sentences:[[...]]},
                  supporting_facts={title:[...], sent_id:[...]}"""
    print("HotpotQA parquet 내려받는 중 (hotpotqa/hotpot_qa, distractor/validation)…")
    rows = _read_parquet_rows("hotpotqa/hotpot_qa",
                              "refs/convert/parquet", "distractor/validation")
    out = []
    for i, r in enumerate(rows):
        if limit and i >= limit:
            break
        ctx = r["context"]                       # {"title":[...], "sentences":[[...]]}
        context = [[t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])]
        sf = r["supporting_facts"]               # {"title":[...], "sent_id":[...]}
        supporting = [[t, int(sid)] for t, sid in zip(sf["title"], sf["sent_id"])]
        out.append({"_id": r.get("id", f"hotpot{i}"),
                    "question": r["question"], "answer": r["answer"],
                    "supporting_facts": supporting, "context": context})
    fp = os.path.join(DATA, "hotpot_dev_distractor_v1.json")
    json.dump(out, open(fp, "w"), ensure_ascii=False)
    print(f"  저장: {fp}  ({len(out)}문항)")
    return fp


# ------------------------------------------------------------ MuSiQue
def fetch_musique(limit=None):
    """MuSiQue-Answerable (validation) parquet → saved as the JSONL that load_musique reads.
    parquet columns: paragraphs=[{idx,title,paragraph_text,is_supporting},...],
                  question, answer, answerable"""
    print("MuSiQue parquet 내려받는 중 (dgslibisey/MuSiQue, validation)…")
    # dgslibisey/MuSiQue is natively parquet → read it directly from the main branch.
    try:
        rows = _read_parquet_rows("dgslibisey/MuSiQue", "main", "valid")
    except Exception:
        rows = _read_parquet_rows("dgslibisey/MuSiQue", "refs/convert/parquet", "valid")
    fp = os.path.join(DATA, "musique_ans_v1.0_dev.jsonl")
    n = 0
    with open(fp, "w") as f:
        for i, r in enumerate(rows):
            if limit and i >= limit:
                break
            paras = [{"idx": p.get("idx", j), "title": p.get("title", ""),
                      "paragraph_text": p.get("paragraph_text", ""),
                      "is_supporting": bool(p.get("is_supporting"))}
                     for j, p in enumerate(r["paragraphs"])]
            rec = {"id": r.get("id", f"musique{i}"),
                   "question": r["question"], "answer": r["answer"],
                   "answerable": bool(r.get("answerable", True)),
                   "paragraphs": paras}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  저장: {fp}  ({n}문항)")
    return fp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hotpotqa", "musique"], default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="문항 수 상한(빠른 확인용). 미지정 시 전량.")
    a = ap.parse_args()
    try:
        if a.only in (None, "hotpotqa"):
            fetch_hotpotqa(a.limit)
        if a.only in (None, "musique"):
            fetch_musique(a.limit)
    except ModuleNotFoundError:
        print("먼저 설치하세요:  pip install -U huggingface_hub pyarrow")
    print("\n완료. 이제:")
    print("  python3 exp1_search.py \\")
    print("      --hotpotqa data/hotpot_dev_distractor_v1.json \\")
    print("      --musique  data/musique_ans_v1.0_dev.jsonl")
