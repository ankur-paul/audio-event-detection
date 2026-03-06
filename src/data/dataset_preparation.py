"""
Dataset preparation pipeline for Audio Event Detection.

Handles:
- Defining the ~47 sound classes
- Creating/loading label CSV files
- Splitting data into train/val/test
- Generating class mappings
"""

import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger()

# 47 environmental sound classes for multi-label audio event detection
DEFAULT_SOUND_CLASSES = [
    # Human vocal sounds (8)
    "speech",
    "laughter",
    "crying",
    "shouting",
    "whispering",
    "singing",
    "cough",
    "sneeze",
    # Human activity sounds (7)
    "footsteps",
    "running",
    "clapping",
    "cheering",
    "applause",
    "breathing",
    "snoring",
    # Household / indoor sounds (8)
    "door_knock",
    "door_close",
    "glass_breaking",
    "keyboard_typing",
    "mouse_click",
    "phone_ringing",
    "alarm_clock",
    "water_running",
    "toilet_flush",
    # Transportation sounds (8)
    "car_engine",
    "car_horn",
    "siren",
    "motorcycle",
    "train",
    "airplane",
    "helicopter",
    "engine_idling",
    # Animal sounds (5)
    "dog_bark",
    "cat_meow",
    "bird_chirping",
    "rooster_crow",
    "cow_moo",
    # Nature sounds (5)
    "rain",
    "thunder",
    "wind",
    "fire_crackling",
    "water_stream",
    # Mechanical / tool sounds (4)
    "hammering",
    "drilling",
    "saw_cutting",
    "machine_running",
    # Impact sounds (1)
    "explosion",
]


def create_class_map(
    classes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Create a mapping from class names to integer indices.

    Args:
        classes: List of class names. Uses DEFAULT_SOUND_CLASSES if None.
        save_path: Path to save the class map as JSON.

    Returns:
        Dictionary mapping class name -> index.
    """
    if classes is None:
        classes = DEFAULT_SOUND_CLASSES

    class_map = {name: idx for idx, name in enumerate(sorted(classes))}

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(class_map, f, indent=2)
        logger.info(f"Class map saved to {save_path} ({len(class_map)} classes)")

    return class_map


def load_class_map(path: str) -> Dict[str, int]:
    """Load class map from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_inverse_class_map(class_map: Dict[str, int]) -> Dict[int, str]:
    """Get index -> class name mapping."""
    return {v: k for k, v in class_map.items()}


def parse_labels_csv(
    csv_path: str,
    delimiter: str = ",",
    label_separator: str = "|",
) -> List[Dict[str, any]]:
    """
    Parse the labels CSV file.

    Expected format:
        filename,labels
        clip001.wav,speech|footsteps
        clip002.wav,rain

    Args:
        csv_path: Path to CSV file.
        delimiter: CSV delimiter.
        label_separator: Separator for multi-labels.

    Returns:
        List of dicts with 'filename' and 'labels' (list of strings).
    """
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            labels = [l.strip() for l in row["labels"].split(label_separator) if l.strip()]
            entries.append({
                "filename": row["filename"].strip(),
                "labels": labels,
            })
    return entries


def labels_to_binary_vector(
    labels: List[str],
    class_map: Dict[str, int],
) -> np.ndarray:
    """
    Convert a list of label strings to a binary vector.

    Args:
        labels: List of class name strings.
        class_map: Mapping from class name to index.

    Returns:
        Binary numpy array of shape (num_classes,).
    """
    num_classes = len(class_map)
    vector = np.zeros(num_classes, dtype=np.float32)
    for label in labels:
        if label in class_map:
            vector[class_map[label]] = 1.0
        else:
            logger.warning(f"Unknown label '{label}' not in class map, skipping.")
    return vector


def split_dataset(
    entries: List[Dict],
    val_split: float = 0.15,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset entries into train/val/test sets.

    Args:
        entries: List of dataset entries.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_entries, val_entries, test_entries).
    """
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_entries = shuffled[:n_test]
    val_entries = shuffled[n_test : n_test + n_val]
    train_entries = shuffled[n_test + n_val :]

    logger.info(
        f"Dataset split: train={len(train_entries)}, "
        f"val={len(val_entries)}, test={len(test_entries)}"
    )

    return train_entries, val_entries, test_entries


def save_split_csv(
    entries: List[Dict],
    save_path: str,
    label_separator: str = "|",
) -> None:
    """Save a list of dataset entries to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "labels"])
        for entry in entries:
            labels_str = label_separator.join(entry["labels"])
            writer.writerow([entry["filename"], labels_str])
    logger.info(f"Saved {len(entries)} entries to {save_path}")
