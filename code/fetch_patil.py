# -*- coding: utf-8 -*-
"""
fetch_patil.py — Downloads the 119 Patil et al. scenario files used in the paper into data/patil/.

The files are taken unmodified from the authors' public repository
    https://github.com/Vaidehi99/MultiAgentPrivacy  (data/experiment_<id>_def_adv_both.json)
and verified against data/patil_manifest.json (MD5 of every file as used in our experiments).
The Patil data are not redistributed in this repository; copyright remains with the original authors.

    python3 fetch_patil.py            # downloads all 119 files (about 2 MB)
    python3 fetch_patil.py --verify   # only verifies files already present

datasets.load_patil() keeps 116 of the 119 scenarios (three are dropped by the loader because the
sensitive combination is missing or covers every cue); see docs/DIALOGUE_RULES.md.
"""
import argparse, hashlib, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "patil")
MANIFEST = os.path.join(HERE, "..", "data", "patil_manifest.json")
RAW = "https://raw.githubusercontent.com/Vaidehi99/MultiAgentPrivacy/main/data/"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    man = json.load(open(MANIFEST))
    os.makedirs(DATA, exist_ok=True)
    bad = 0
    for i, (name, want) in enumerate(sorted(man.items()), 1):
        fp = os.path.join(DATA, name)
        if not os.path.exists(fp):
            if a.verify:
                print("missing:", name); bad += 1; continue
            urllib.request.urlretrieve(RAW + name, fp)
        got = md5(fp)
        if got != want:
            print(f"MD5 mismatch: {name} (got {got}, expected {want})"); bad += 1
        if i % 20 == 0:
            print(f"  ...{i}/{len(man)}")
    print(f"{len(man) - bad}/{len(man)} files OK in {os.path.abspath(DATA)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
