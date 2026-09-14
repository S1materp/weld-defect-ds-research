# Welding Defect Segmentation — Research Project

Research-oriented computer vision project for **welding defect segmentation in radiographic images** using **YOLOv8n-seg**.

The project focuses on controlled experimentation rather than training a single model: I compare augmentation strategies and image resolutions, analyze quality/speed trade-offs, investigate class imbalance, and run a **paper-inspired augmentation experiment** based on ideas from research on automated welding-defect detection.

---

## Research Questions

The project investigates three main questions:

1. Does stronger data augmentation improve welding-defect segmentation?
2. Does increasing input resolution from 416 to 640 improve segmentation quality?
3. Can a paper-inspired augmentation strategy improve the strongest experimental configuration?

---

## Dataset

Dataset:

`viacheslavasadchiy/radiographs-welding-defect-detection`

Full dataset:

- 21,436 training images
- 5,360 validation images
- 13 defect classes

For reproducible and Colab-friendly experiments, I created a fixed research subset:

- 796 training images
- 198 validation images
- random seed: `42`

The same subset was used across all experiments.

---

## Model

All experiments use:

- **YOLOv8n-seg**
- pretrained weights
- AdamW optimizer
- learning rate `1e-3`
- 10 epochs
- fixed random seed `42`
- identical train/validation split
- NVIDIA Tesla T4
- segmentation metrics

The main evaluation metric is **mask mAP50-95**.

---

## Experiments

Four controlled configurations were evaluated.

| Experiment | Resolution | Main Change |
|---|---:|---|
| `baseline_416` | 416 | Baseline configuration |
| `augmented_416` | 416 | Stronger augmentation |
| `highres_640` | 640 | Increased image resolution |
| `paper_aug_640` | 640 | Paper-inspired stronger augmentation |

---

## Results

| Experiment | Precision | Recall | mAP50 | mAP50-95 | Inference |
|---|---:|---:|---:|---:|---:|
| `baseline_416` | 0.3815 | 0.2857 | 0.2586 | 0.2334 | 20.18 ms |
| `augmented_416` | 0.2469 | 0.2773 | 0.2739 | 0.2479 | 19.74 ms |
| `highres_640` | 0.4306 | **0.3112** | **0.2963** | 0.2655 | 27.08 ms |
| **`paper_aug_640`** | **0.6720** | 0.2792 | 0.2862 | **0.2721** | 27.54 ms |

### mAP50-95

![mAP50-95 comparison](results/mAP50-95_comparison.png)

### mAP50

![mAP50 comparison](results/mAP50_comparison.png)

### Precision

![Precision comparison](results/precision_comparison.png)

### Recall

![Recall comparison](results/recall_comparison.png)

### Inference Speed

![Inference speed comparison](results/inference_speed_comparison.png)

---

## Experiment 1 — Baseline

The first experiment established the reference performance at 416px.

**Result:**

`mAP50-95 = 0.2334`

This value was used as the main baseline for subsequent experiments.

---

## Experiment 2 — Stronger Augmentation

The second experiment tested whether stronger augmentation could improve generalization while keeping the same 416px resolution.

mAP50-95 increased:

`0.2334 → 0.2479`

This corresponds to approximately **+6.2% relative improvement** over the baseline.

The result suggested that augmentation could be beneficial, but also showed that different metrics did not improve uniformly.

---

## Experiment 3 — Higher Resolution

The third experiment increased input resolution:

`416 → 640`

mAP50-95 increased:

`0.2334 → 0.2655`

This is approximately **+13.8% relative improvement** over the baseline.

However, inference time increased:

`20.18 ms/image → 27.08 ms/image`

This demonstrates a clear **quality vs. inference-cost trade-off**.

At this stage, `highres_640` became the strongest configuration.

---

## Paper-Inspired Experiment

After the first three experiments, I explored research on deep-learning-based welding-defect detection.

A recurring idea in welding-defect research is the use of stronger data augmentation to improve generalization under limited and imbalanced defect data.

Instead of claiming a full paper reproduction, I adapted this idea to the existing segmentation pipeline and designed an additional controlled experiment.

### Hypothesis

> A stronger paper-inspired augmentation strategy applied at 640px will improve segmentation quality over the existing `highres_640` configuration.

### Augmentation Strategy

The `paper_aug_640` experiment uses:

- Mosaic
- MixUp
- geometric transformations
- translation
- scaling
- horizontal flipping
- stronger intensity/color augmentation

The model architecture, dataset split, seed, optimizer, learning rate and number of epochs were kept consistent.

### Result

`highres_640`:

`mAP50-95 = 0.265546`

`paper_aug_640`:

`mAP50-95 = 0.272064`

Relative improvement:

**+2.45%**

Therefore, the hypothesis was supported for the main segmentation metric.

However, the improvement was **not uniform across all metrics**.

Precision increased substantially:

`0.4306 → 0.6720`

while recall decreased:

`0.3112 → 0.2792`

and mAP50 decreased slightly:

`0.2963 → 0.2862`

This indicates a trade-off rather than universal improvement.

---

## Overall Improvement

The best configuration improved mask mAP50-95 from:

`0.233440 → 0.272064`

Compared with the original baseline, this is approximately:

**+16.5% relative improvement**

The experimental progression was:

`baseline_416 → augmented_416 → highres_640 → paper_aug_640`

---

## Class Imbalance Analysis

![Class distribution](results/class_distribution.png)

The reduced research subset is strongly class-imbalanced.

This became one of the main limitations identified during experimentation.

For the final `paper_aug_640` model, some defect classes performed very well:

- `defect_6`: mask mAP50 ≈ **0.912**
- `defect_7`: mask mAP50 ≈ **0.972**

At the same time, several rare classes had near-zero performance.

This suggests that aggregate metrics alone do not fully describe model behavior.

---

## Failure Analysis

The experiments revealed several important failure modes:

- rare defect classes are poorly represented;
- performance differs substantially between classes;
- increasing resolution improves quality but increases inference cost;
- aggressive augmentation can improve mAP50-95 while reducing recall;
- metrics for classes with only a few validation examples are unstable.

Therefore, class-aware evaluation is important for future experiments.

---

## Example Predictions

Qualitative predictions are available here:

[`results/example_predictions`](results/example_predictions)

These examples can be used to visually inspect segmentation quality and model failure cases.

---

## Limitations

The current study has several limitations:

- a reduced subset of the full dataset was used;
- the subset is strongly class-imbalanced;
- experiments were performed with one random seed;
- training was limited to 10 epochs;
- only one model family was evaluated;
- rare-class metrics have high uncertainty;
- the paper-inspired experiment is an adaptation of research ideas, not a full reproduction of a published architecture.

---

## Next Research Steps

Possible extensions:

1. Implement class-aware or stratified subset sampling.
2. Run experiments across multiple random seeds.
3. Report mean and standard deviation across runs.
4. Perform detailed per-class AP analysis.
5. Analyze false-positive and false-negative examples.
6. Compare another segmentation architecture.
7. Reproduce and adapt a specific architecture-level method from a scientific paper.
8. Evaluate the best configuration on a larger or full dataset.

---

## Repository Structure

```text
weld-defect-ds-research/
├── README.md
├── requirements.txt
├── src/
│   └── Weld_Defect_DS_Research_FINAL.py
├── docs/
│   └── research_summary.txt
└── results/
    ├── experiment_results.csv
    ├── paper_experiment_summary.txt
    ├── class_distribution.csv
    ├── class_distribution.png
    ├── mAP50-95_comparison.png
    ├── mAP50_comparison.png
    ├── precision_comparison.png
    ├── recall_comparison.png
    ├── inference_speed_comparison.png
    └── example_predictions/
```

---

## Reproduction

Experiments were run in **Google Colab with an NVIDIA Tesla T4 GPU**.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main experiment pipeline:

```bash
python src/Weld_Defect_DS_Research_FINAL.py
```

Experiment artifacts and metrics are stored in the `results/` directory.

---

## Tech Stack

**Python · PyTorch · YOLOv8 · Ultralytics · OpenCV · Pandas · NumPy · Matplotlib · CUDA · Google Colab · Git**

---

## Author

**Serafima Saltanova**

GitHub: `@S1materp`