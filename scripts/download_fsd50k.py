"""
Download and prepare FSD50K dataset.

FSD50K: Freesound Dataset 50K
- ~51,000 clips, 200 classes (AudioSet ontology)
- Download from Zenodo (free, no registration for download)
- Size: ~30 GB total (dev + eval sets)

Source: https://zenodo.org/record/4060432

NOTE: FSD50K is large. This script downloads it in parts and can resume
interrupted downloads. We primarily use FSD50K.dev_audio and FSD50K.eval_audio.

Usage:
    python scripts/download_fsd50k.py
    python scripts/download_fsd50k.py --dev-only     # skip eval set to save time
    python scripts/download_fsd50k.py --metadata-only # just download labels
"""

import argparse
import csv
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.label_mapping import FSD50K_MAPPING, map_labels, get_coverage_report

# Zenodo download URLs for FSD50K
# Record: https://zenodo.org/record/4060432
FSD50K_FILES = {
    "dev_audio": [
        ("FSD50K.dev_audio.z01", "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01"),
        ("FSD50K.dev_audio.z02", "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z02"),
        ("FSD50K.dev_audio.z03", "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z03"),
        ("FSD50K.dev_audio.z04", "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z04"),
        ("FSD50K.dev_audio.z05", "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z05"),
        ("FSD50K.dev_audio.zip", "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip"),
    ],
    "eval_audio": [
        ("FSD50K.eval_audio.z01", "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01"),
        ("FSD50K.eval_audio.zip", "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip"),
    ],
    "metadata": [
        ("FSD50K.ground_truth.zip", "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip"),
        ("FSD50K.metadata.zip", "https://zenodo.org/records/4060432/files/FSD50K.metadata.zip"),
    ],
}


def download_file(url: str, dest_path: str) -> None:
    """Download a file with progress, supporting resume."""
    dest = Path(dest_path)

    # Check if already downloaded
    if dest.exists():
        # Verify by checking content-length if available
        print(f"  Already downloaded: {dest.name}")
        return

    print(f"  Downloading: {dest.name}")
    print(f"    URL: {url}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    Progress: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        else:
            mb = downloaded / (1024 * 1024)
            print(f"\r    Downloaded: {mb:.1f} MB", end="", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = str(dest) + ".partial"

    try:
        urllib.request.urlretrieve(url, partial, reporthook=_progress)
        os.rename(partial, str(dest))
        print(f"\n    Complete: {dest.name}")
    except Exception as e:
        print(f"\n    ERROR downloading {dest.name}: {e}")
        if os.path.exists(partial):
            os.remove(partial)
        raise


def download_fsd50k(output_dir: str, dev_only: bool = False, metadata_only: bool = False) -> str:
    """
    Download FSD50K dataset files.

    Args:
        output_dir: Download directory.
        dev_only: Only download dev set (skip eval).
        metadata_only: Only download metadata/ground truth.

    Returns:
        Path to download directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always download metadata
    print("\n=== Downloading FSD50K metadata ===")
    for filename, url in FSD50K_FILES["metadata"]:
        download_file(url, str(output_dir / filename))

    # Extract metadata
    for filename, _ in FSD50K_FILES["metadata"]:
        zip_path = output_dir / filename
        if zip_path.exists():
            print(f"  Extracting {filename}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_dir)

    if metadata_only:
        print("Metadata-only mode. Skipping audio download.")
        return str(output_dir)

    # Download dev audio
    print("\n=== Downloading FSD50K dev audio (~22 GB) ===")
    print("This is a multi-part zip. All parts are needed.")
    for filename, url in FSD50K_FILES["dev_audio"]:
        download_file(url, str(output_dir / filename))

    # Extract dev audio (multi-part zip - extract from the .zip file)
    dev_audio_dir = output_dir / "FSD50K.dev_audio"
    if not dev_audio_dir.exists():
        main_zip = output_dir / "FSD50K.dev_audio.zip"
        if main_zip.exists():
            print("Extracting dev audio (this may take a while)...")
            # For multi-part zips, we need the 'zip' command
            os.system(f'cd "{output_dir}" && zip -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio_combined.zip && unzip FSD50K.dev_audio_combined.zip')

    if not dev_only:
        # Download eval audio
        print("\n=== Downloading FSD50K eval audio (~8 GB) ===")
        for filename, url in FSD50K_FILES["eval_audio"]:
            download_file(url, str(output_dir / filename))

        eval_audio_dir = output_dir / "FSD50K.eval_audio"
        if not eval_audio_dir.exists():
            main_zip = output_dir / "FSD50K.eval_audio.zip"
            if main_zip.exists():
                print("Extracting eval audio...")
                os.system(f'cd "{output_dir}" && zip -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio_combined.zip && unzip FSD50K.eval_audio_combined.zip')

    return str(output_dir)


def prepare_fsd50k_labels(fsd50k_dir: str, output_csv: str, include_eval: bool = True) -> int:
    """
    Read FSD50K ground truth and produce labels in our unified format.

    FSD50K ground truth files:
    - FSD50K.ground_truth/dev.csv  (columns: fname, labels, mids, split)
    - FSD50K.ground_truth/eval.csv (columns: fname, labels, mids)

    Labels are comma-separated AudioSet ontology labels.

    Returns:
        Number of clips with matched labels.
    """
    gt_dir = os.path.join(fsd50k_dir, "FSD50K.ground_truth")

    if not os.path.exists(gt_dir):
        print(f"ERROR: Ground truth directory not found: {gt_dir}")
        print("Run with --metadata-only first, then download audio.")
        return 0

    entries = []

    # Process dev set
    dev_csv = os.path.join(gt_dir, "dev.csv")
    if os.path.exists(dev_csv):
        dev_entries = _process_fsd50k_csv(
            dev_csv,
            audio_dir=os.path.join(fsd50k_dir, "FSD50K.dev_audio"),
            source_tag="fsd50k_dev",
        )
        entries.extend(dev_entries)
        print(f"  FSD50K dev: {len(dev_entries)} matched clips")

    # Process eval set
    if include_eval:
        eval_csv = os.path.join(gt_dir, "eval.csv")
        if os.path.exists(eval_csv):
            eval_entries = _process_fsd50k_csv(
                eval_csv,
                audio_dir=os.path.join(fsd50k_dir, "FSD50K.eval_audio"),
                source_tag="fsd50k_eval",
            )
            entries.extend(eval_entries)
            print(f"  FSD50K eval: {len(eval_entries)} matched clips")

    if not entries:
        print("WARNING: No labels were matched. Check ground truth files exist.")
        return 0

    # Write output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "labels", "source", "source_path"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"\nFSD50K preparation complete:")
    print(f"  Total matched clips: {len(entries)}")
    print(f"  Output: {output_csv}")

    report = get_coverage_report(FSD50K_MAPPING)
    print(f"  Coverage: {report['covered_count']}/{report['total']} "
          f"classes ({report['coverage_pct']:.0f}%)")
    print(f"  Missing: {', '.join(report['missing'])}")

    return len(entries)


def _process_fsd50k_csv(csv_path: str, audio_dir: str, source_tag: str) -> list:
    """Process a single FSD50K ground truth CSV."""
    entries = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["fname"]
            source_labels = [l.strip() for l in row["labels"].split(",")]

            target_labels = map_labels(source_labels, FSD50K_MAPPING)
            if not target_labels:
                continue

            audio_file = f"{fname}.wav"
            audio_path = os.path.join(audio_dir, audio_file)

            entries.append({
                "filename": f"fsd50k_{fname}.wav",
                "labels": "|".join(target_labels),
                "source": source_tag,
                "source_path": audio_path,
            })

    return entries


def main():
    parser = argparse.ArgumentParser(description="Download and prepare FSD50K")
    parser.add_argument(
        "--output-dir", type=str, default="data/downloads/fsd50k",
        help="Directory to download FSD50K to.",
    )
    parser.add_argument(
        "--labels-csv", type=str, default="data/downloads/fsd50k_labels.csv",
        help="Path to save prepared labels CSV.",
    )
    parser.add_argument(
        "--dev-only", action="store_true",
        help="Only download/prepare dev set.",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Only download metadata (ground truth labels), skip audio.",
    )
    args = parser.parse_args()

    fsd50k_dir = download_fsd50k(
        args.output_dir,
        dev_only=args.dev_only,
        metadata_only=args.metadata_only,
    )
    prepare_fsd50k_labels(
        fsd50k_dir,
        args.labels_csv,
        include_eval=not args.dev_only,
    )


if __name__ == "__main__":
    main()
