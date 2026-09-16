# -*- coding: utf-8 -*-
"""
================================================================================
 fetch_datasets.py (v2) — HotpotQA · MuSiQue 를 HuggingFace parquet 로 직접 받아
 exp1_search.py 의 로더가 그대로 읽는 원본 포맷 파일로 저장한다.
================================================================================
 왜 parquet 직접?
   - `datasets.load_dataset` 는 Python 3.14 에서 내부 pickling(dill/tqdm)이 깨지고,
     hotpotqa/hotpot_qa 는 '스크립트 기반'이라 최신 datasets 가 로드를 거부한다.
   - HF 는 모든 데이터셋을 parquet 로 자동 변환(refs/convert/parquet)해 두므로,
     그 parquet 만 huggingface_hub 로 받아서 pyarrow 로 읽으면 위 문제를 모두 우회한다.
     (무거운 datasets 런타임/트러스트코드 불필요 → 3.14 에서도 동작)

 준비:  pip install -U huggingface_hub pyarrow
 실행:  cd hpd && python3 fetch_datasets.py            # 둘 다
        python3 fetch_datasets.py --only hotpotqa
        python3 fetch_datasets.py --only musique --limit 500

 이후:  python3 exp1_search.py \
            --hotpotqa data/hotpot_dev_distractor_v1.json \
            --musique  data/musique_ans_v1.0_dev.jsonl
================================================================================
"""
import os, json, argparse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


def _read_parquet_rows(repo_id, revision, want):
    """repo_id 의 parquet 파일 중 경로에 want(예: 'validation')를 포함하는 것들을
    모두 받아 행(dict) 리스트로 반환. datasets 라이브러리 없이 huggingface_hub+pyarrow 사용."""
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
    """hotpotqa/hotpot_qa (distractor, validation) parquet → 원본 JSON 스키마로 저장.
    parquet 컬럼: context={title:[...], sentences:[[...]]},
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
    """MuSiQue-Answerable(validation) parquet → load_musique 가 읽는 JSONL 로 저장.
    parquet 컬럼: paragraphs=[{idx,title,paragraph_text,is_supporting},...],
                  question, answer, answerable"""
    print("MuSiQue parquet 내려받는 중 (dgslibisey/MuSiQue, validation)…")
    # dgslibisey/MuSiQue 는 parquet 네이티브 → main 브랜치에서 바로 읽는다.
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
