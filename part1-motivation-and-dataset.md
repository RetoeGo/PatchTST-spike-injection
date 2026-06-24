---
title: "Does Patch Averaging Make PatchTST More Robust to Training Anomalies? Part 1: Motivation and Control Dataset"
date: 2026-06-24
---

# Part 1: Motivation and Control Dataset

> This is Part 1 of a two-part post. Part 1 covers *why* this experiment matters and *what* control dataset we built to test it. [Part 2](./part2-experiments-and-findings.md) covers the experimental setup and results.

## The property we are testing

[PatchTST](https://arxiv.org/abs/2211.14730) (Nie et al., 2023, ICLR) changed how Transformers are applied to time series forecasting by splitting each input series into patches of `P` consecutive timesteps (default `P = 16`) and embedding each patch as a single token, instead of giving every individual timestep its own token. The paper motivates this design around three benefits: retaining local semantic information, reducing the quadratic cost of attention, and allowing the model to attend over a longer history. Patch averaging is explicitly positioned as a representational choice for capturing patterns, not as a robustness mechanism.

A vanilla Transformer for forecasting — following [Vaswani et al., 2017, "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — does the opposite: every timestep gets its own token, so the attention mechanism sees and can directly weight each individual value.

This asymmetry suggested a property the PatchTST paper does not test:

> **Property under test:** Because PatchTST compresses 16 raw timesteps into one embedding vector, a single anomalous timestep injected during training can only ever contribute a fraction (≈1/16, when the anomaly is narrow) of that vector. A vanilla Transformer, by contrast, gives the same anomalous timestep a full, dedicated token. We predicted PatchTST should therefore generalize better than a vanilla Transformer to a *clean* test set after training on data containing a single localized anomaly, and that this advantage should depend on how wide the anomaly is relative to the patch length (16 hours).

This is a robustness claim about patch-based tokenization that sits one inferential step beyond what either source paper measures directly — PatchTST evaluates accuracy on clean benchmarks, and the standard Transformer baseline is evaluated under the same clean conditions. Neither paper trains on corrupted data and checks generalization to clean data afterward. That gap is what our control dataset is built to fill.

## Why a control dataset, and not just "noisy data"

To isolate *this specific mechanism* (patch averaging dilutes anomalies) from everything else that differs between PatchTST and a vanilla Transformer (different parameter counts, different inductive biases, different training dynamics), the dataset has to satisfy four constraints:

1. **Exactly one anomaly, nothing else changed.** If we added many anomalies or general noise, any MAE difference between models could come from the model's general noise-handling rather than from the patch-averaging mechanism specifically. One injected anomaly means one cause for any observed effect.
2. **The test set must stay clean.** The question is generalization *after* training on a corrupted distribution — not robustness to anomalies present at test time. If the test set were also corrupted, we'd be measuring something else (in-distribution anomaly handling, not out-of-distribution generalization).
3. **The anomaly's width must be a controlled variable, swept across the patch boundary.** The property predicts the PatchTST advantage depends on anomaly width relative to the 16-hour patch length. A dataset with a single fixed-width anomaly couldn't show whether the effect is architecture-general or width-dependent — we need a width sweep that brackets 16 hours from well below to well above it.
4. **The anomaly must be invertible in sign.** Injecting a large positive value shifts the mean and standard deviation that get fit by the `StandardScaler` used in both pipelines. That shift alone — independent of any architecture — could change test MAE. Testing both a positive and a negative spike of the same shape lets us check this: if normalization shift were driving the result, the negative spike should *hurt* test MAE (since it would shift the scaler in the opposite, presumably "wrong," direction), while the positive spike helps. If both directions show a similar pattern, the effect is more likely architectural.

Each of these is a specific design choice with a specific failure mode it rules out — not just "make the data realistic."

## The base dataset

We use [ETTh1](https://github.com/zhouhaoyi/ETDataset) (Electricity Transformer Temperature, hourly) — a standard long-term time-series forecasting benchmark with 17,420 hourly records across 7 variables, used as one of the primary benchmarks in the PatchTST paper itself. The target variable is `OT` (oil temperature). We use the standard 70/10/20 train/validation/test split (8,640 / 2,880 / 2,880 hours), the same split used by both PatchTST and the original Transformer forecasting baselines.

Using a benchmark dataset that PatchTST is already known to perform well on means any behavioral difference we observe under anomaly injection is attributable to the anomaly, not to PatchTST simply being a poor fit for this data.

## How the anomaly is generated

We inject a single Gaussian "spike" into the `OT` column of the training and validation portions only — never into the test set:

```
X'(t) = X(t) + A · exp(−(t − t₀)² / 2σ²)
```

- **t₀** is fixed at the midpoint of the training window, far from the train/test boundary, so the spike can't bleed into or interact with boundary effects in how the windows are constructed.
- **A = 10 × (95th percentile of `OT` in training)** — large enough that the spike is unambiguously visible above the normal range of the signal, not something that could be mistaken for ordinary variation.
- **σ (the spike's width)** is swept across six values: **1, 4, 16, 32, 64, and 128 hours.** This range is chosen relative to PatchTST's patch length of 16: at σ=1 the spike is essentially a single anomalous point (much narrower than one patch); at σ=16 it spans almost exactly one patch; at σ=128 it spans roughly eight patches.

A Gaussian shape is used because σ is the *only* parameter that changes shape — there's no discontinuity at the edges (as a box-shaped spike would have), so changing σ purely changes "how many timesteps are affected" without introducing new shape artifacts.

For the falsification check, we generate a mirrored **negative-spike** version of every σ condition:

```
X'(t) = X(t) − A · exp(−(t − t₀)² / 2σ²)
```

### Generation code

The full injection logic, run directly on the downloaded ETTh1 CSV:

```python
import numpy as np
import pandas as pd

ETT_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
ETT_TARGET = "OT"

df_ett = pd.read_csv("dataset/ETTh1.csv")

BORDER2_TRAIN = 12 * 30 * 24                      # end of train  = 8640
BORDER1_TEST  = 12 * 30 * 24 + 4 * 30 * 24        # start of test = 11520

n  = len(df_ett)
t  = np.arange(n, dtype=float)
t0 = int(BORDER2_TRAIN * 0.5)                     # midpoint of training window

p95_train = np.percentile(df_ett[ETT_TARGET].iloc[:BORDER2_TRAIN].values, 95)
A = 10.0 * p95_train

SIGMA_VARIANTS = [
    ("sigma_1",    1),
    ("sigma_4",    4),
    ("sigma_16",  16),
    ("sigma_32",  32),
    ("sigma_64",  64),
    ("sigma_128", 128),
]

for label, sigma in SIGMA_VARIANTS:
    S = A * np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    df_variant = df_ett.copy()
    # Only training + validation rows (index < BORDER1_TEST) are modified.
    # Test rows are left completely untouched.
    df_variant.loc[df_variant.index < BORDER1_TEST, ETT_TARGET] = (
        df_variant.loc[df_variant.index < BORDER1_TEST, ETT_TARGET]
        + S[:BORDER1_TEST]
    )
    df_variant.to_csv(f"dataset/ETTh1_{label}.csv", index=False)
```

The negative-spike variants are generated identically, with the sign of `S` flipped before it's added.

## What the control dataset looks like

| σ (hours) | Spike width (FWHM) | Fraction of train+val affected | Patches touched (out of 16h each) |
|---|---|---|---|
| 1   | ~2 hours    | <0.1% | 1–2 patches, heavily diluted |
| 4   | ~9 hours    | ~0.1% | ~1 patch |
| 16  | ~38 hours   | ~0.4% | ~3 patches |
| 32  | ~75 hours   | ~0.9% | ~5 patches |
| 64  | ~150 hours  | ~1.7% | ~10 patches |
| 128 | ~300 hours  | ~3.5% | ~20 patches |

At σ=1, the spike is a single sharp point that is essentially invisible at the scale of the full 8,640-hour training window — you would not spot it by eye on a full-series plot. At σ=128, it is a broad, smooth hump spanning several hundred hours, clearly visible even when zoomed out. Everywhere in between, the spike sits inside the otherwise-real ETTh1 signal, so the model is still learning genuine seasonal and daily oil-temperature patterns — it just has one corrupted region to contend with.

Below is the actual injected signal across all six σ values, overlaid on the original clean `OT` series (train/validation region only — the clean test region, in green, is never touched):

*(See `etth1_spike_variants.png` in the repo for the rendered figure — generated by the plotting cell in the notebook, showing the original OT series with all six σ variants overlaid in the train/val window, with the test region shaded green and a single-patch-width reference band shaded gold around the spike center.)*

## Dataset and code availability

- **Generation code (full notebook):** `[github link — e.g. https://github.com/<your-username>/<your-repo>/blob/main/patchtst_etth1_spike.ipynb]`
- **Generated CSVs** (`ETTh1_sigma_1.csv` … `ETTh1_sigma_128.csv`, plus `_neg` variants): `[github link — e.g. https://github.com/<your-username>/<your-repo>/tree/main/dataset]`
- **Base dataset (ETTh1, unmodified):** [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
- **PatchTST reference implementation used for training:** [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)

[Part 2](./part2-experiments-and-findings.md) covers the model setup, training configuration, and what we actually found.
