"""
Download and prepare UrbanSound8K dataset.

UrbanSound8K:
- 8732 clips, 10 classes, <=4 seconds each
- Requires agreeing to terms at the download page

NOTE: UrbanSound8K cannot be auto-downloaded due to license terms.
You must manually download it from:
    https://urbansounddataset.weebly.com/urbansound8k.html

Steps:
1. Visit the URL above
2. Click the download link and agree to terms
3. Extract the archive to data/downloads/urbansound8k/
4. Run this script to prepare labels

Usage:
    python scripts/download_urbansound8k.py
    python scripts/download_urbansound8k.py --input-dir /path/to/UrbanSound8K
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.label_mapping import (
    URBANSOUND8K_MAPPING,
    URBANSOUND8K_CLASS_ID_TO_NAME,
    map_labels,
    get_coverage_report,
)


def prepare_urbansound8k_labels(us8k_dir: str, output_csv: str) -> int:
    """
    Read UrbanSound8K metadata and produce labels in our unified format.

    US8K metadata CSV has columns:
    slice_file_name, fsID, start, end, salience, fold, classID, class

    Audio is in: audio/fold{1-10}/

    Returns:
        Number of clips with matched labels.
    """
    meta_path = os.path.join(us8k_dir, "metadata", "UrbanSound8K.csv")

    if not os.path.exists(meta_path):
        print(f"ERROR: UrbanSound8K metadata not found at {meta_path}")
        print()
        print("UrbanSound8K requires manual download due to license terms.")
        print("Steps:")
        print("  1. Visit: https://urbansounddataset.weebly.com/urbansound8k.html")
        print("  2. Download and extract the archive")
        print(f"  3. Place it at: {us8k_dir}/")
        print(f"     Expected structure:")
        print(f"       {us8k_dir}/metadata/UrbanSound8K.csv")
        print(f"       {us8k_dir}/audio/fold1/ ... fold10/")
        print(f"  4. Re-run this script")
        return 0

    entries = []
    skipped = 0

    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row["classID"])
            class_name = URBANSOUND8K_CLASS_ID_TO_NAME.get(class_id, "")
            target_labels = map_labels([class_name], URBANSOUND8K_MAPPING)

            if not target_labels:
                skipped += 1
                continue

            audio_file = row["slice_file_name"]
            fold = row["fold"]
            audio_path = os.path.join(us8k_dir, "audio", f"fold{fold}", audio_file)

            if not os.path.exists(audio_path):
                continue

            entries.append({
                "filename": f"us8k_{audio_file}",
                "labels": "|".join(target_labels),
                "source": "urbansound8k",
                "source_path": audio_path,
            })

    # Write output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "labels", "source", "source_path"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"\nUrbanSound8K preparation complete:")
    print(f"  Matched clips: {len(entries)}")
    print(f"  Skipped clips: {skipped} (no matching class)")
    print(f"  Output: {output_csv}")

    report = get_coverage_report(URBANSOUND8K_MAPPING)
    print(f"  Coverage: {report['covered_count']}/{report['total']} "
          f"classes ({report['coverage_pct']:.0f}%)")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Prepare UrbanSound8K labels")
    parser.add_argument(
        "--input-dir", type=str, default="data/downloads/urbansound8k/UrbanSound8K",
        help="Path to extracted UrbanSound8K directory.",
    )
    parser.add_argument(
        "--labels-csv", type=str, default="data/downloads/urbansound8k_labels.csv",
        help="Path to save prepared labels CSV.",
    )
    args = parser.parse_args()

    prepare_urbansound8k_labels(args.input_dir, args.labels_csv)


if __name__ == "__main__":
    main()
