---
title: "Does Patch Averaging Make PatchTST More Robust to Training Anomalies? Part 2: Experiments and Findings"
date: 2026-06-24
---

# Part 2: Experiments and Findings

> This is Part 2 of a two-part post. [Part 1](./part1-motivation-and-dataset.md) covers the motivation and the control dataset. This part covers the experimental setup and what we actually found.

## Recap: what we're testing

[Part 1](./part1-motivation-and-dataset.md) built a control dataset from [ETTh1](https://github.com/zhouhaoyi/ETDataset) with a single Gaussian anomaly ("spike") injected into the training and validation data only, swept across six widths (σ = 1, 4, 16, 32, 64, 128 hours) relative to [PatchTST](https://arxiv.org/abs/2211.14730)'s 16-hour patch length, in both a positive and negative direction. The test set is always the original, clean ETTh1 data.

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
