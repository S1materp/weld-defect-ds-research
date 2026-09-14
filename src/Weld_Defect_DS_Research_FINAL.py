# Welding Defect Segmentation — DS Research Project
# FINAL Colab version: saves everything directly to Google Drive.
#
# Colab:
# 1) Runtime -> Change runtime type -> T4 GPU
# 2) Run:
#    !pip install -q ultralytics kagglehub opencv-python pyyaml pandas matplotlib tabulate
# 3) Upload this .py to /content
# 4) Run:
#    !python /content/Weld_Defect_DS_Research_FINAL.py
#
# All important outputs are written to:
# /content/drive/MyDrive/Weld_Defect_DS_Research

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import kagglehub
from ultralytics import YOLO
from google.colab import drive

DRIVE_MOUNT = Path("/content/drive")
if not (DRIVE_MOUNT / "MyDrive").exists():
    print("Mounting Google Drive...")
    drive.mount(str(DRIVE_MOUNT))

PROJECT_ROOT = DRIVE_MOUNT / "MyDrive" / "Weld_Defect_DS_Research"
DATA_SMALL = PROJECT_ROOT / "data_small"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

for p in [PROJECT_ROOT, DATA_SMALL, RUNS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Saving project to:", PROJECT_ROOT)

try:
    shutil.copy2(Path(__file__).resolve(), PROJECT_ROOT / Path(__file__).name)
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("Seed:", SEED)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU is not enabled.")

DATASET_HANDLE = "viacheslavasadchiy/radiographs-welding-defect-detection"

print("\nDownloading / locating dataset...")
dataset_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
data_root = dataset_root / "data_kaggle"

train_images_dir = data_root / "train" / "images"
train_labels_dir = data_root / "train" / "labels"
val_images_dir = data_root / "val" / "images"
val_labels_dir = data_root / "val" / "labels"

for p in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    if not p.exists():
        raise FileNotFoundError(f"Expected dataset path not found: {p}")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"

def detect_class_ids(label_dirs: List[Path]) -> List[int]:
    ids = set()
    for label_dir in label_dirs:
        for txt in label_dir.glob("*.txt"):
            try:
                for line in txt.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if parts:
                        ids.add(int(float(parts[0])))
            except Exception:
                pass
    return sorted(ids)

train_images = list_images(train_images_dir)
val_images = list_images(val_images_dir)

print("Train images:", len(train_images))
print("Val images:", len(val_images))

class_ids = detect_class_ids([train_labels_dir, val_labels_dir])
if not class_ids:
    raise RuntimeError("Could not detect class ids.")

NUM_CLASSES = max(class_ids) + 1
CLASS_NAMES = [f"defect_{i}" for i in range(NUM_CLASSES)]

print("Detected class ids:", class_ids)
print("Number of classes:", NUM_CLASSES)

TRAIN_SUBSET_SIZE = min(800, len(train_images))
VAL_SUBSET_SIZE = min(200, len(val_images))

subset_rng = random.Random(SEED)
selected_train = subset_rng.sample(train_images, TRAIN_SUBSET_SIZE)
selected_val = subset_rng.sample(val_images, VAL_SUBSET_SIZE)

def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    reset_dir(DATA_SMALL / split / "images")
    reset_dir(DATA_SMALL / split / "labels")

def copy_pairs(images, source_labels_dir, target_images_dir, target_labels_dir):
    copied = 0
    for img in images:
        lbl = label_path_for(img, source_labels_dir)
        if not lbl.exists():
            continue
        shutil.copy2(img, target_images_dir / img.name)
        shutil.copy2(lbl, target_labels_dir / lbl.name)
        copied += 1
    return copied

copied_train = copy_pairs(
    selected_train,
    train_labels_dir,
    DATA_SMALL / "train" / "images",
    DATA_SMALL / "train" / "labels",
)
copied_val = copy_pairs(
    selected_val,
    val_labels_dir,
    DATA_SMALL / "val" / "images",
    DATA_SMALL / "val" / "labels",
)

print("Research subset:", copied_train, "train,", copied_val, "val")

def class_distribution(labels_dir: Path) -> Dict[int, int]:
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for txt in labels_dir.glob("*.txt"):
        try:
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                counts[cls] = counts.get(cls, 0) + 1
        except Exception:
            pass
    return counts

train_dist = class_distribution(DATA_SMALL / "train" / "labels")
val_dist = class_distribution(DATA_SMALL / "val" / "labels")

dist_df = pd.DataFrame({
    "class_id": list(range(NUM_CLASSES)),
    "class_name": CLASS_NAMES,
    "train_instances": [train_dist.get(i, 0) for i in range(NUM_CLASSES)],
    "val_instances": [val_dist.get(i, 0) for i in range(NUM_CLASSES)],
})
dist_df.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)

plt.figure(figsize=(11, 5))
x = np.arange(NUM_CLASSES)
width = 0.4
plt.bar(x - width / 2, dist_df["train_instances"], width, label="train")
plt.bar(x + width / 2, dist_df["val_instances"], width, label="val")
plt.xticks(x, CLASS_NAMES, rotation=45)
plt.ylabel("Instances")
plt.title("Class distribution in research subset")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=160)
plt.close()

data_yaml = {
    "path": str(DATA_SMALL),
    "train": "train/images",
    "val": "val/images",
    "nc": NUM_CLASSES,
    "names": CLASS_NAMES,
}

yaml_path = PROJECT_ROOT / "data_small.yaml"
with yaml_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

EXPERIMENTS = [
    {
        "name": "baseline_416",
        "imgsz": 416,
        "epochs": 10,
        "batch": 8,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.20,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "mosaic": 0.0,
    },
    {
        "name": "augmented_416",
        "imgsz": 416,
        "epochs": 10,
        "batch": 8,
        "degrees": 7.0,
        "translate": 0.15,
        "scale": 0.60,
        "fliplr": 0.5,
        "hsv_h": 0.02,
        "hsv_s": 0.80,
        "hsv_v": 0.50,
        "mosaic": 1.0,
    },
    {
        "name": "highres_640",
        "imgsz": 640,
        "epochs": 10,
        "batch": 4,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.20,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "mosaic": 0.0,
    },
]

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")

def extract_metrics(metrics):
    out = {
        "precision": float("nan"),
        "recall": float("nan"),
        "map50": float("nan"),
        "map50_95": float("nan"),
    }
    seg = getattr(metrics, "seg", None)
    if seg is not None:
        out["precision"] = safe_float(getattr(seg, "mp", np.nan))
        out["recall"] = safe_float(getattr(seg, "mr", np.nan))
        out["map50"] = safe_float(getattr(seg, "map50", np.nan))
        out["map50_95"] = safe_float(getattr(seg, "map", np.nan))
        return out
    box = getattr(metrics, "box", None)
    if box is not None:
        out["precision"] = safe_float(getattr(box, "mp", np.nan))
        out["recall"] = safe_float(getattr(box, "mr", np.nan))
        out["map50"] = safe_float(getattr(box, "map50", np.nan))
        out["map50_95"] = safe_float(getattr(box, "map", np.nan))
    return out

results_rows = []

for exp in EXPERIMENTS:
    print("\n" + "=" * 72)
    print("RUNNING:", exp["name"])
    print("=" * 72)

    exp_dir = RUNS_DIR / exp["name"]
    weights_path = exp_dir / "weights" / "best.pt"

    if weights_path.exists():
        print("Found existing best.pt on Drive -> skip training.")
        best_model = YOLO(str(weights_path))
        train_seconds = float("nan")
    else:
        model = YOLO("yolov8n-seg.pt")
        train_start = time.perf_counter()

        model.train(
            data=str(yaml_path),
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            workers=2,
            device=DEVICE,
            seed=SEED,
            deterministic=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=1e-3,
            project=str(RUNS_DIR),
            name=exp["name"],
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
            degrees=exp["degrees"],
            translate=exp["translate"],
            scale=exp["scale"],
            fliplr=exp["fliplr"],
            hsv_h=exp["hsv_h"],
            hsv_s=exp["hsv_s"],
            hsv_v=exp["hsv_v"],
            mosaic=exp["mosaic"],
        )

        train_seconds = time.perf_counter() - train_start

        if not weights_path.exists():
            raise FileNotFoundError(f"best.pt not found for {exp['name']}")

        best_model = YOLO(str(weights_path))

    val_start = time.perf_counter()
    metrics = best_model.val(
        data=str(yaml_path),
        split="val",
        imgsz=exp["imgsz"],
        batch=exp["batch"],
        device=DEVICE,
        plots=True,
        verbose=True,
        project=str(RUNS_DIR),
        name=f"{exp['name']}_val",
    )
    val_seconds = time.perf_counter() - val_start

    m = extract_metrics(metrics)

    val_sample = list_images(DATA_SMALL / "val" / "images")[:50]
    infer_start = time.perf_counter()
    _ = best_model.predict(
        source=[str(p) for p in val_sample],
        imgsz=exp["imgsz"],
        conf=0.25,
        device=DEVICE,
        verbose=False,
    )
    infer_seconds = time.perf_counter() - infer_start
    ms_per_image = 1000 * infer_seconds / max(len(val_sample), 1)

    row = {
        "experiment": exp["name"],
        "imgsz": exp["imgsz"],
        "epochs": exp["epochs"],
        "batch": exp["batch"],
        "precision": m["precision"],
        "recall": m["recall"],
        "mAP50": m["map50"],
        "mAP50-95": m["map50_95"],
        "train_minutes": train_seconds / 60 if not np.isnan(train_seconds) else np.nan,
        "val_seconds": val_seconds,
        "inference_ms_per_image": ms_per_image,
        "weights": str(weights_path),
    }

    results_rows.append(row)

    pd.DataFrame(results_rows).to_csv(
        RESULTS_DIR / "experiment_results_partial.csv",
        index=False,
    )

    print(pd.Series(row))
    print("Partial results saved to Google Drive.")

results_df = pd.DataFrame(results_rows)
results_df.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)

print("\nFINAL RESULTS")
print(
    results_df[
        [
            "experiment",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "inference_ms_per_image",
        ]
    ].round(4)
)

for metric in ["precision", "recall", "mAP50", "mAP50-95"]:
    plt.figure(figsize=(8, 4))
    plt.bar(results_df["experiment"], results_df[metric])
    plt.ylabel(metric)
    plt.xlabel("Experiment")
    plt.title(f"{metric}: experiment comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{metric}_comparison.png", dpi=160)
    plt.close()

plt.figure(figsize=(8, 4))
plt.bar(results_df["experiment"], results_df["inference_ms_per_image"])
plt.ylabel("ms / image")
plt.xlabel("Experiment")
plt.title("Inference speed comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "inference_speed_comparison.png", dpi=160)
plt.close()

best_idx = results_df["mAP50-95"].astype(float).idxmax()
best_row = results_df.loc[best_idx]

print("\nBest experiment by mAP50-95:")
print(best_row)

best_model = YOLO(best_row["weights"])
example_images = list_images(DATA_SMALL / "val" / "images")[:8]

best_model.predict(
    source=[str(p) for p in example_images],
    imgsz=int(best_row["imgsz"]),
    conf=0.25,
    save=True,
    project=str(RESULTS_DIR),
    name="example_predictions",
    exist_ok=True,
    device=DEVICE,
    verbose=False,
)

baseline_row = results_df.loc[
    results_df["experiment"] == "baseline_416"
].iloc[0]

summary = f"""
Welding Defect Segmentation — Research Summary
==============================================

Research question:
How do controlled data augmentation and input image resolution affect
YOLOv8n-seg performance for welding-defect segmentation on radiographic images?

Protocol:
- fixed random seed: {SEED}
- same train/validation subset for all experiments
- same model family: YOLOv8n-seg
- offline validation
- metrics: Precision, Recall, mAP50, mAP50-95, inference time
- artifacts saved directly to Google Drive

Best experiment:
- configuration: {best_row["experiment"]}
- mAP50-95: {float(best_row["mAP50-95"]):.4f}
- mAP50: {float(best_row["mAP50"]):.4f}
- precision: {float(best_row["precision"]):.4f}
- recall: {float(best_row["recall"]):.4f}
- inference: {float(best_row["inference_ms_per_image"]):.2f} ms/image

Baseline:
- mAP50-95: {float(baseline_row["mAP50-95"]):.4f}
- inference: {float(baseline_row["inference_ms_per_image"]):.2f} ms/image

Important limitation:
The sampled validation set is strongly class-imbalanced. Metrics for rare
classes can be unstable, so per-class conclusions must be interpreted carefully.

Other limitations:
- reduced subset for Colab-friendly runtime;
- one random seed;
- one model family;
- short 10-epoch training schedule.

Next steps:
- analyze per-class AP and failure cases;
- improve the split strategy;
- run multiple seeds and report mean/std;
- compare another segmentation architecture;
- reproduce/adapt one method from a relevant scientific paper.
"""

(RESULTS_DIR / "research_summary.txt").write_text(summary, encoding="utf-8")

(PROJECT_ROOT / "experiment_config.json").write_text(
    json.dumps(
        {
            "seed": SEED,
            "dataset": DATASET_HANDLE,
            "train_subset_size": copied_train,
            "val_subset_size": copied_val,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "experiments": EXPERIMENTS,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(summary)
print("\nDONE.")
print("Everything is saved in:")
print(PROJECT_ROOT)
