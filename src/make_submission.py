
#!/usr/bin/env python3
"""
make_submission.py
──────────────────
Post-process the CSVs from beats_knn_asd.py so they match DCASE-2025
submission rules, then zip everything into <run_name>.zip.

Run *after* you have called beats_knn_asd.py infer for every machine.
"""
import csv, glob, os, shutil, zipfile, yaml, argparse, pathlib
import numpy as np
from scipy.stats import gamma

def gamma_threshold(errs, pct: float = 90.0) -> float:
    a, loc, scale = gamma.fit(errs, floc=0)
    return float(gamma.ppf(pct / 100.0, a, loc=loc, scale=scale))

def postprocess(scores_csv: str, out_dir: str, gamma_pct: float = 90.0):
    machine = pathlib.Path(scores_csv).stem.split("_")[0]          # e.g. ToyCar
    section = "00"                                                 # Task-2 dev/eval only has section 00
    # 1) read anomaly scores
    names, scores = [], []
    with open(scores_csv) as f:
        reader = csv.reader(f)
        header = next(reader)              # discard header
        for name, s in reader:
            names.append(name)
            scores.append(float(s))
    scores = np.array(scores)
    # 2) threshold
    thr = gamma_threshold(scores, gamma_pct)
    decisions = (scores > thr).astype(int)
    # 3) write header-less files in required format
    anom_file = f"anomaly_score_{machine}_section_{section}_test.csv"
    decid_file = f"decision_result_{machine}_section_{section}_test.csv"
    with open(os.path.join(out_dir, anom_file), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(names, scores))
    with open(os.path.join(out_dir, decid_file), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(names, decisions))
    print(f"✓ {machine}: wrote {anom_file}, {decid_file} (thr={thr:.4f})")

def main():
    P = argparse.ArgumentParser()
    P.add_argument("--scores-glob", default="*_scores.csv",
                   help="pattern that matches ALL per-machine score CSVs")
    P.add_argument("--out-dir", default="submission_files")
    P.add_argument("--run-name", default="run1")
    args = P.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for csv_path in glob.glob(args.scores_glob):
        postprocess(csv_path, args.out_dir)

    # --- minimal meta file (edit to taste) --------------------------
    meta = {
        "task": "DCASE2025 Task 2",
        "system": args.run_name,
        "feature_extractor": "BEATs-base",
        "detector": "k-NN (k=1, cosine)",
        "threshold": "Gamma {}th-pct on training scores".format(90),
    }
    with open(os.path.join(args.out_dir, f"{args.run_name}_meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f)

    # --- zip everything --------------------------------------------
    zip_path = f"{args.run_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pathlib.Path(args.out_dir).iterdir():
            zf.write(p, arcname=p.name)
    print(f"\n🎉  Created submission archive → {zip_path}")

if __name__ == "__main__":
    main()