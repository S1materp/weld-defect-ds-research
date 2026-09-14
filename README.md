# Welding Defect Segmentation — DS Research

Research-oriented computer vision project for **welding defect segmentation on radiographic images** using **YOLOv8n-seg**.

The goal of the project is not only to train a model, but to compare controlled experimental configurations and analyze the trade-off between segmentation quality and inference speed.

## Research Question

**How do stronger data augmentation and increased input resolution affect YOLOv8n-seg performance for welding-defect segmentation?**

## Dataset

Dataset: `viacheslavasadchiy/radiographs-welding-defect-detection`

Full dataset:
- Train images: **21,436**
- Validation images: **5,360**

For Colab-friendly controlled experiments, a fixed reproducible subset was used:
- Train: **796 image-label pairs**
- Validation: **198 image-label pairs**
- Random seed: **42**

The validation subset is strongly class-imbalanced, so results for rare classes should be interpreted cautiously.

## Experimental Setup

All experiments use:
- **YOLOv8n-seg**
- pretrained weights
- AdamW optimizer
- learning rate `1e-3`
- 10 epochs
- fixed seed `42`
- the same train/validation subset
- offline evaluation on the same validation split

Three configurations were compared:

1. **baseline_416** — 416 px input with light geometric augmentation
2. **augmented_416** — 416 px input with stronger augmentation
3. **highres_640** — 640 px input with baseline-style augmentation

## Results

| Experiment | Precision | Recall | mAP50 | mAP50-95 | Inference, ms/image |
|---|---:|---:|---:|---:|---:|
| baseline_416 | 0.3815 | 0.2857 | 0.2586 | 0.2334 | 20.18 |
| augmented_416 | 0.2469 | 0.2773 | 0.2739 | 0.2479 | 19.74 |
| **highres_640** | **0.4306** | **0.3112** | **0.2963** | **0.2655** | 27.08 |

## Key Findings

The **high-resolution 640 px configuration** achieved the best overall segmentation quality:

- `mAP50-95`: **0.2655**
- `mAP50`: **0.2963**
- Precision: **0.4306**
- Recall: **0.3112**

Compared with the baseline, increasing the input resolution improved `mAP50-95` from **0.2334 to 0.2655**, an approximately **13.8% relative improvement**.

The trade-off was inference speed:
- baseline: **20.18 ms/image**
- high-resolution: **27.08 ms/image**

The stronger augmentation configuration also improved `mAP50-95` over baseline:
- baseline: **0.2334**
- augmented: **0.2479**
- relative improvement: approximately **6.2%**

## Failure Analysis

Performance differs strongly across classes.

The model performs substantially better on well-represented classes such as `defect_6` and `defect_7`, while several rare classes have near-zero metrics.

For the high-resolution model:
- `defect_6`: mask mAP50 ≈ **0.948**
- `defect_7`: mask mAP50 ≈ **0.951**

At the same time, some rare classes contain only one or two validation instances.

This suggests that **class imbalance and limited rare-class coverage are major bottlenecks** in the current experimental setup.

## Limitations

- Experiments were run on a reduced subset rather than the full dataset.
- The validation subset is strongly class-imbalanced.
- Only one random seed was used.
- Training was limited to 10 epochs.
- Only one model family was compared.
- Rare-class conclusions are not statistically robust.

## Next Research Steps

- design a more class-aware train/validation sampling strategy;
- evaluate per-class AP in more detail;
- perform qualitative failure analysis;
- run several seeds and report mean/std;
- compare another segmentation architecture;
- reproduce or adapt a method from a relevant scientific paper.

## Repository Structure

```text
weld-defect-ds-research/
├── README.md
├── src/
│   └── Weld_Defect_DS_Research_FINAL.py
├── results/
│   ├── experiment_results.csv
│   ├── class_distribution.csv
│   ├── mAP50-95_comparison.png
│   ├── mAP50_comparison.png
│   ├── precision_comparison.png
│   ├── recall_comparison.png
│   ├── inference_speed_comparison.png
│   └── example_predictions/
├── docs/
│   └── research_summary.txt
├── requirements.txt
└── .gitignore
```

## Run in Google Colab

```bash
!pip install -q ultralytics kagglehub opencv-python pyyaml pandas matplotlib tabulate
!python /content/Weld_Defect_DS_Research_FINAL.py
```

The training script stores artifacts directly in Google Drive:

```text
/content/drive/MyDrive/Weld_Defect_DS_Research
```

## Tech Stack

Python, PyTorch, Ultralytics YOLOv8, OpenCV, Pandas, NumPy, Matplotlib, KaggleHub, Google Colab, CUDA.

## Author

Serafima Saltanova
