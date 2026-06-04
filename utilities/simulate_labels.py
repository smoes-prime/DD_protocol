#!/usr/bin/env python3
"""
HASTEN-style simulated docking for Deep Docking:
look up precomputed scores instead of running Glide/FRED.

Writes training_labels.txt (every iteration) and, on iteration 1,
validation_labels.txt and testing_labels.txt from valid_set.txt / test_set.txt.

Usage:
  python utilities/simulate_labels.py \\
    --master_scores /home/sebmo/HASTEN_proj/DD_data/master_scores/1B12_div_docked_DD_labels.csv \\
    --project_dir /home/sebmo/HASTEN_proj/DD_data/projects/1B12 \\
    --iteration 1
"""

import argparse
import os
import sys


def load_scores(path):
    scores = {}
    with open(path) as f:
        header = f.readline()
        if "ZINC_ID" not in header and "zinc" not in header.lower():
            # file had no header; first line is data
            parts = header.strip().split(",")
            if len(parts) >= 2:
                scores[parts[-1].strip()] = float(parts[0])
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            score = float(parts[0])
            zinc_id = parts[-1].strip()
            scores[zinc_id] = score
    return scores


def read_id_list(path):
    ids = []
    with open(path) as f:
        for line in f:
            zid = line.strip().split(",")[0].strip()
            if zid:
                ids.append(zid)
    return ids


def write_labels(out_path, ids, scores, missing_out):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    missing = []
    with open(out_path, "w") as out:
        out.write("r_i_docking_score,ZINC_ID\n")
        for zid in ids:
            if zid not in scores:
                missing.append(zid)
                continue
            out.write(f"{scores[zid]},{zid}\n")
    return missing


def main():
    p = argparse.ArgumentParser(description="Simulate DD docking from a master score file")
    p.add_argument("--master_scores", required=True, help="CSV: score,ZINC_ID with header")
    p.add_argument("--project_dir", required=True, help="e.g. .../DD_data/projects/1B12")
    p.add_argument("--iteration", type=int, required=True)
    args = p.parse_args()

    it_dir = os.path.join(args.project_dir, f"iteration_{args.iteration}")
    if not os.path.isdir(it_dir):
        print(f"Missing iteration directory: {it_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading master scores...")
    scores = load_scores(args.master_scores)
    print(f"  {len(scores)} scores loaded")

    train_ids = read_id_list(os.path.join(it_dir, "train_set.txt"))
    missing_all = []

    missing = write_labels(
        os.path.join(it_dir, "training_labels.txt"),
        train_ids,
        scores,
        missing_all,
    )
    missing_all.extend(missing)
    print(f"training_labels.txt: {len(train_ids) - len(missing)} / {len(train_ids)} IDs")

    if args.iteration == 1:
        for split, fname in [("validation", "valid_set.txt"), ("testing", "test_set.txt")]:
            id_path = os.path.join(it_dir, fname)
            if not os.path.isfile(id_path):
                print(f"Warning: missing {id_path}", file=sys.stderr)
                continue
            ids = read_id_list(id_path)
            missing = write_labels(
                os.path.join(it_dir, f"{split}_labels.txt"),
                ids,
                scores,
                missing_all,
            )
            missing_all.extend(missing)
            print(f"{split}_labels.txt: {len(ids) - len(missing)} / {len(ids)} IDs")

    if missing_all:
        miss_file = os.path.join(it_dir, "simulate_labels_missing.txt")
        with open(miss_file, "w") as f:
            f.write("\n".join(missing_all[:50000]))
            if len(missing_all) > 50000:
                f.write(f"\n... and {len(missing_all) - 50000} more\n")
        print(
            f"WARNING: {len(missing_all)} IDs had no score in master file. "
            f"First 50k listed in {miss_file}. Continuing anyway.",
            file=sys.stderr,
        )

    print("Done.")


if __name__ == "__main__":
    main()
