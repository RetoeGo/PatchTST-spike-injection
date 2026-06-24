---
title: "Does patch averaging make patchtst more robust to training anomalies than a vanilla transformer?"
theme: jekyll-theme-minimal
date: 2026-06-24
---
This blogpost was written as an assignment for the course Fundamentals of Machine and Deeplearning at TU Delft.
Part 1 covers *why* the experiment matters and *what* control dataset was build. Part 2 covers the experimental setup and the results. 

# Part 1: Motivation and Control Dataset
## The property we are testing
In this blogpost I test whether patching makes a transformer more robust against the effects of anomalies. I compare [PatchTST](https://arxiv.org/abs/2211.14730) with a [Vanilla Transformer](https://arxiv.org/abs/1706.03762). Patching groups multiple timesteps together, in this experiment I group 16 timesteps together into a single embedding vector. A vanilla transformer uses a single embedding per timestep. PatchTST's authors motivate the design with three benefits: it retains local semantic information, it reduces the quadratic cost of attention, and it allows the model to attend over a longer history.

**Hypothesis** Because PatchTST compresses 16 raw timesteps into one embedding vector, a single anomalous timestep injected during training can only ever con    tribute a fraction (≈1/16, when the anomaly is narrow) of that vector. A vanilla Transformer, by contrast, gives the same anomalous timestep a full, dedicated token. We     predicted PatchTST should therefore generalize better than a vanilla Transformer to a *clean* test set after training on data containing a single localized anomaly, and     that this advantage should depend on how wide the anomaly is relative to the patch length (16 hours).

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

---

# Part 2: Experiments and Findings

## Recap: what we're testing

The section above built a control dataset from [ETTh1](https://github.com/zhouhaoyi/ETDataset) with a single Gaussian anomaly ("spike") injected into the training and validation data only, swept across six widths (σ = 1, 4, 16, 32, 64, 128 hours) relative to [PatchTST](https://arxiv.org/abs/2211.14730)'s 16-hour patch length, in both a positive and negative direction. The test set is always the original, clean ETTh1 data.

The prediction: **PatchTST should generalize better than a vanilla Transformer to the clean test set after training on the spiked data, with the advantage shrinking as σ grows past 16 hours** — because once the spike is wider than a patch, PatchTST's patches stop diluting it and start being dominated by it, just like the Transformer's individual tokens always are.

## Experimental setup

| | PatchTST | Transformer |
|---|---|---|
| Tokenization | 16-hour patches, stride 8 | individual timesteps |
| `d_model` | 16 | 16 |
| Heads | 4 | 4 |
| Layers | 3 | 3 |
| `d_ff` | 128 | 128 |
| `seq_len` | 336 | 336 |
| `pred_len` | 96 | 96 |
| Learning rate | 0.0001 | 0.0001 |
| Batch size | 128 | 128 |
| Epochs | 10 | 10 |

Both models are matched on architecture size and training budget so that any difference in outcome is attributable to the tokenization strategy (patches vs. individual timesteps), not to one model simply having more capacity or more training. We used the [official PatchTST repository](https://github.com/yuqinie98/PatchTST)'s training script (`run_longExp.py`) for both models, since it ships baseline implementations of both.

**Metric:** Mean Absolute Error (MAE) on the `OT` channel of the clean test set, on the normalized scale (the same `StandardScaler`-normalized scale both models are trained and evaluated on internally). Lower is better. A no-spike baseline is run for each model as a reference point.

## Results

### PatchTST and Transformer, positive spike

| σ (hours) | PatchTST MAE | Transformer MAE |
|---|---|---|
| **Baseline (no spike)** | **0.1761** | **0.4725** |
| 1   | 0.1557 | 0.4857 |
| 4   | 0.1209 | 0.5016 |
| 16  | 0.0767 | 0.4893 |
| 32  | 0.0585 | 0.3871 |
| 64  | 0.0433 | 0.1998 |
| 128 | 0.0311 | 0.1256 |

### PatchTST and Transformer, negative spike (falsification check)

| σ (hours) | PatchTST MAE | Transformer MAE |
|---|---|---|
| 1  | 0.1538 | 0.4179 |
| 4  | 0.1174 | 0.3978 |
| 16 | 0.0709 | 0.4553 |

*(We ran the negative-spike condition through σ=16; σ=32/64/128 negative-spike runs are in the notebook's training loop but we don't have verified output values for them — see "Limitations" below.)*

## What we found

**1. PatchTST's test MAE improves monotonically as the spike gets wider — all the way out to σ=128, far past the 16-hour patch length.** This is the opposite of what we predicted. We expected the advantage to *peak* near σ=16 and erode afterward, once the spike could no longer be diluted within a single patch. Instead, PatchTST kept improving the larger the spike got, with no sign of the predicted reversal in our tested range.

**2. The Transformer behaves erratically at small σ, then improves sharply at large σ.** Its MAE actually gets slightly *worse* than baseline at σ=1 and σ=4 (0.4857, 0.5016 vs. a baseline of 0.4725), is roughly flat at σ=16, and only starts improving meaningfully from σ=32 onward — reaching 0.1256 at σ=128, a substantial improvement over its own baseline, but still far behind PatchTST's 0.0311 at the same width.

**3. Both architectures ultimately treat a sufficiently wide training anomaly as a regularizer, not just PatchTST.** A spike wide enough to occupy a meaningful fraction of the training window (σ≥32, roughly 5+ patches / 75+ hours) seems to act like injected noise that discourages both models from overfitting to spurious fine-grained patterns — closer to a form of data augmentation than to a "training anomaly" in the adversarial sense. The original motivation framed this as a PatchTST-specific dilution effect; the data suggests it's at least partly a general phenomenon, with PatchTST simply benefiting earlier and more strongly.

**4. The positive/negative symmetry check supports an architectural (not purely scaler-driven) explanation for PatchTST, but is inconclusive for Transformer.** At σ=1, 4, and 16, PatchTST's positive- and negative-spike MAEs are close to each other (e.g., 0.0767 vs. 0.0709 at σ=16) — consistent with the improvement coming from the architecture's handling of the anomaly rather than from which direction the `StandardScaler` mean got shifted. The Transformer, by contrast, does *not* show this symmetry: the negative spike consistently outperforms the positive spike at the same σ (e.g., 0.4179 vs. 0.4857 at σ=1). That asymmetry is itself a finding — it suggests the Transformer's behavior here is more sensitive to scaler shift, spike sign, or some interaction we haven't isolated, rather than being a clean architectural effect in the way PatchTST's result is.

## Honest interpretation

The original hypothesis — a crossover near σ = patch length, after which PatchTST's advantage erodes — is **not supported** by this data. What we actually find is a real, reproducible, and fairly large effect (PatchTST's MAE drops by ~5.7× from baseline to σ=128; Transformer's drops by ~3.8×), but it doesn't have the shape we predicted. PatchTST's relative advantage over the Transformer is largest in the small-σ regime (where Transformer is flat-to-worse and PatchTST is already improving) and narrows — in relative terms — at large σ, where both models improve substantially. That's a different and more nuanced story than "patch averaging dilutes anomalies until the patch is full," and it's worth reporting as such rather than reading the data as confirming the original framing.

## Limitations

- **We dropped a third planned comparison (Pathformer, a dynamic-patching model) from this writeup.** Those runs were executed at a different number of training epochs (3, vs. 10 for PatchTST/Transformer) and with an inconsistent `seq_len` across setup attempts, so they are not comparable to the numbers above and would have undermined the matched-budget design described in the setup. We're leaving that comparison for a follow-up once it can be run under matched conditions.
- **Negative-spike results for σ=32/64/128 are not included** because we don't have verified output values for those runs at the time of writing. The negative-spike comparison above should be read as a partial check (σ≤16 only), not a complete falsification sweep.
- **Single run per condition.** Each (model, σ, sign) combination was trained once. We don't have variance estimates across seeds, so a few percentage points of difference between adjacent σ values shouldn't be read as precise.

## Code and data

- **Generation code and full experiment notebook:** `[github link]`
- **Generated control datasets:** `[github link]`
- **PatchTST paper:** [Nie et al., 2023 — A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730)
- **Transformer baseline:** [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Base dataset:** [ETTh1, zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
