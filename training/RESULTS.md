# Local training results

## Test-split looks (log every one)

| date | what was looked at | by |
|---|---|---|
| 2026-11-23 | public baseline seeds 42 and 7 vs v1.0, and s42 vs s7 (held-out match, 396 frames) | Claude session |
| 2026-11-23 | experiment #4: P2 head and plain fine-tune vs baseline, seeds 42 and 7 | Claude session |
| 2026-11-23 | data-quality experiment: no-unreviewed and human-only training sets vs baseline, seeds 42 and 7 | Claude session |

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

## Experiment #4: P2 (stride-4) head (2026-11-23), rejected

Two arms, each trained with seeds 42 and 7:
- **P2:** `yolo26s-p2.yaml` (an extra 240×240 grid at stride 4), COCO `yolo26s.pt` weights loaded where shapes match (766/902 tensors), with `--trainer plain`.
- **Control:** `yolo26s.pt` with `--trainer plain`.

Same recipe as the baseline otherwise. P2 trains in 16.3 min per seed (7.5 GiB), against 12.3 min for the standard model.

**The control turned out identical to the baseline.** Ultralytics 8.4.153 has `cls_remap=True` by default, which maps pretrained classes to the dataset's by name: "Transferred 708/708 items", COCO's sports-ball row included. Deterministic training then reproduced the baseline exactly (identical test detections; `best.pt` differs only in metadata). So with this Ultralytics version, the author's `SportsBallTransferTrainer` and the default trainer give the same model.

Highest val mAP50 over the 30 epochs (not necessarily the selected checkpoint): baseline/control 0.743 (s42), 0.734 (s7); P2 0.720 (s42), 0.704 (s7).

Held-out match, P2 − baseline at the same seed (95 % CI):

| | seed 42 | seed 7 |
|---|---|---|
| maxTP@FP≤25 | −4 [−15, +7] | −6 [−16, +4] |
| maxTP@FP≤60 | +0 [−12, +11] | −4 [−14, +4] |
| maxTP@FP≤100 | −5 [−16, +7] | −11 [−22, +1] |
| maxTP@FP≤200 | −7 [−19, +3] | −6 [−16, +4] |
| smallTP@0.25 | −17 [−27, −8] | −14 [−22, −6] |
| TP/FP at 0.25 (P2 vs baseline) | 94/52 vs 114/153 | 92/88 vs 108/134 |

**Verdict: no gain.** At equal false-positive budgets P2 is equal or worse in every cell for both seeds. At fixed thresholds it is less confident and finds fewer small balls. On top of that it costs 1.3× the training time and more compute on the phone. The finer grid does not recover the missed small balls here, which is consistent with the earlier finding that the limit is generalising to new matches, not resolution. Added to the negative results.

## Error analysis on val (2026-11-23, no test look)

`analysis/error_analysis.py`, confidence 0.25, val (292 frames, 279 balls). Each false positive and each missed ball is classed (rules in the script).

| | base_s42 | v1.0 |
|---|---|---|
| false positives | 51 | 45 |
| … isolated (real false alarms) | 23 (45 %) | 28 (62 %) |
| … near-miss (on a ball, 0.1 ≤ IoU < 0.5) | 20 (39 %) | 13 (29 %) |
| … static (same spot in ≥ 3 other frames) | 7 (14 %) | 3 (7 %) |
| … near ball / duplicate | 1 / 0 | 1 / 0 |
| missed balls | 80 | 72 |
| … no candidate at all | 48 (60 %) | 42 (58 %) |
| … misplaced (a box above threshold, IoU < 0.5) | 17 (21 %) | 11 (15 %) |
| … weak (only below-threshold boxes nearby) | 15 (19 %) | 19 (26 %) |

What it means:
- **Hard negatives (plan #6) are not worth doing yet.** Static objects cause 3 to 7 false positives.
- **About a third of the "false positives" are boxes on a real ball** that miss IoU 0.5, which is harsh for 8–16 px balls. Most pair with a "misplaced" miss, so one slightly-off box counts as both a false positive and a miss. For aiming a camera, the ball's centre matters more than its box. A centre-distance metric (hit if the centre is within one ball width) would measure that better. That's an evaluation change to agree with the author before adopting it.
- **60 % of misses have no candidate at all,** which again points to data and generalisation rather than tuning.

## Data-quality experiment (2026-11-23)

The public train split mixes three labelling methods (`metadata.csv`, `label_method`): 1,074 human-reviewed frames, 477 auto-labelled frames with only the uncertain ones reviewed (`auto_judge_reviewed`), and 122 unreviewed auto-labelled frames (`pitch_session_2`, `auto_judge_unreviewed`). Same recipe, 2 seeds, val unchanged:
- **no_unreviewed:** 1,551 frames (drops the 122 unreviewed).
- **human_only:** 1,074 frames (also drops the 477 auto-labelled).

Held-out match, arm − baseline at the same seed, max TP at FP ≤ 25 / 60 / 100 / 200 (95 % CI):

| | seed 42 | seed 7 |
|---|---|---|
| no_unreviewed | −1 [−12, +10] / +5 [−7, +17] / +2 [−9, +14] / −2 [−14, +9] | −8 [−23, +1] / −2 [−15, +8] / −9 [−19, +2] / −6 [−16, +3] |
| human_only | **−14 [−23, −4] / −13 [−23, −4] / −11 [−21, −1] / −13 [−25, −3]** | −7 [−17, +2] / −6 [−18, +2] / **−14 [−24, −2] / −11 [−22, −2]** |

**Verdict:**
- **The 122 unreviewed frames don't hurt:** no significant change. Keep them.
- **Removing the 477 auto-labelled frames clearly hurts,** with both seeds lower at every budget and 6 of 8 CIs excluding zero.
- **Model-proposed labels with review of the uncertain ones add real value.** That's direct support for the pseudo-labelling pipeline in `teacher/` and for more data as the main lever.
