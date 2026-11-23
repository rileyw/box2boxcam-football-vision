# Local training results

## Test-split looks (log every one)

| date | what was looked at | by |
|---|---|---|
| 2026-11-23 | public baseline seeds 42 and 7 vs v1.0, and s42 vs s7 (held-out match, 396 frames) | Claude session |

## Public-data baseline, 2 seeds (2026-11-23)

The v1.0 recipe (`train_local.py`, COCO YOLO26s, sports-ball transfer, 960 px, batch 8, 30 epochs, AdamW 0.001, cosine, mosaic 0, scale 0.3, AMP, deterministic) was trained on the **public train split only** (1,673 images). Checkpoints were picked on public val, which comes from the same match as train.

- **Training time on the RTX 5070 Ti:** 12.3 min per seed, all 30 epochs, peak 5.3 GiB.
- **Best epoch on val:** 19 for seed 42 (mAP50 0.743) and 28 for seed 7 (0.712).

### Held-out match (public test split: 396 frames, 219 balls, 149 small)

Linux, CUDA, `evaluate_detector.py` at 960 px, IoU ≥ 0.5. Cells read TP/FP, with small TP in brackets.

| model | 0.25 | 0.35 | 0.45 | 0.55 | max TP at FP ≤ 25/60/100/200 |
|---|---|---|---|---|---|
| baseline s42 | 114/153 (70) | 104/85 (63) | 97/43 (58) | 86/13 (48) | 93 / 97 / 106 / 118 |
| baseline s7 | 108/134 (66) | 105/94 (65) | 94/62 (56) | 82/14 (45) | 88 / 94 / 107 / 115 |
| v1.0 (released) | 100/104 (56) | 93/52 (51) | 85/21 (46) | 82/4 (46) | 85 / 93 / 100 / 110 |

Paired bootstrap, A − B with 95 % CI:

| | s42 − v1.0 | s7 − v1.0 | s42 − s7 |
|---|---|---|---|
| TP@0.25 | +14 [+3, +26] | +8 [−1, +18] | +6 [−5, +18] |
| FP@0.25 | +49 [+26, +70] | +30 [+8, +52] | +19 [−1, +40] |
| smallTP@0.25 | +14 [+5, +24] | +10 [+3, +18] | +4 [−5, +14] |
| FP@0.45 | +22 [+8, +35] | +41 [+26, +56] | −19 [−32, −6] |
| maxTP@FP≤25 | +8 [−4, +17] | +3 [−7, +12] | +5 [−6, +14] |
| maxTP@FP≤60 | +4 [−6, +16] | +1 [−9, +12] | +3 [−6, +13] |
| maxTP@FP≤100 | +6 [−5, +19] | +7 [−4, +18] | −1 [−11, +10] |
| maxTP@FP≤200 | +8 [−5, +19] | +5 [−5, +14] | +3 [−8, +15] |

### Reading

- **The public-only baseline is not below v1.0, contrary to the plan's expectation.**
  - At equal false-positive budgets, both seeds match it: every difference is +1 to +8, and every CI includes zero.
  - Both seeds are more confident. At a fixed threshold they find more balls (small balls +10 to +14 at 0.25, with CIs excluding zero) and make more false positives.
  - So the baseline sits at a different operating point on roughly the same precision-recall curve.
  - About half the data appears to cost nothing on this match. The extra private v1.0 data is mostly third-party and TV frames plus auto-labels, which may not help on amateur footage.
- **Seed noise is large at fixed thresholds and small at equal budgets.**
  - The two seeds differ by 19 false positives at 0.45 (CI [−32, −6]).
  - Their budget differences stay within ±5 (CIs about ±10).
  - Experiments should therefore be judged on the **max TP at FP budget** rows, with at least 2 seeds. Fixed-threshold rows mostly reflect calibration.
- **What it means for the work ahead.** The public baseline is a fair stand-in for v1.0 as the reference. Any experiment must beat roughly +10 TP at a fixed FP budget to clear this noise. On one test match with 219 balls, that is a high bar, which supports filming more matches (plan step 2).

Files:
- weights in `eval/models/`, copied from `runs/base_s*/weights/best.pt`
- reports in `eval/*_vs_*.json`
- raw detections cached in `eval/cache/`
