"""M10: merge the per-object result CSVs of the single-object models into one BOP results file.

Each single-object job (scripts/m10_single.sh) writes a CSV with only its object's poses. BOP
needs one file per method, with one time per image shared by all estimates of that image: here
the time of an image is the sum of the times of the models that ran on it (running all of them).

Usage:
  .venv/bin/python scripts/m10_merge.py OUT_CSV IN_CSV...
"""
import csv
import sys
from collections import defaultdict


def main(out_path, in_paths):
    rows, image_time = [], defaultdict(float)
    for path in in_paths:
        times = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                times[(row["scene_id"], row["im_id"])] = float(row["time"])
        for key, t in times.items():
            image_time[key] += t
    with open(out_path, "w", newline="") as f:
        fields = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"]))):
            row["time"] = image_time[(row["scene_id"], row["im_id"])]
            writer.writerow(row)
    print(f"{len(rows)} estimates from {len(in_paths)} files, {len(image_time)} images -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
