# Core ML parity of box2boxcam v1.0

Research of 23 November 2026. It asks whether the model the iPhone app actually runs, `box2boxcam-ball-v1.0.mlpackage`, behaves like `box2boxcam-ball-v1.0.pt`, the PyTorch model that the README's results describe.

## Summary

- **The Core ML export finds the same balls, but returns about 1.8× the false positives.** At confidence 0.25 it gives 96 true positives and 184 false positives, against 99 / 100 for the `.pt`. The difference is +84 false positives, with a 95 % CI of [+50, +119].
- **Precision, compute units and the network conversion are not the cause.** An fp32 export still gives +74 false positives. CPU and Neural Engine behave alike. The converted network reproduces PyTorch's raw scores exactly.
- **The cause is the NMS stage of the Core ML pipeline.** Ultralytics passes the standard IoU threshold (0.7) to Apple's `NonMaximumSuppression` model, which interprets it differently: at 0.7 it keeps near-duplicate boxes that a standard NMS at 0.7 removes, and behaves like a standard NMS at about 0.9. At 0.45 it matches a standard NMS at 0.7.
- **Each of three small fixes brings the Core ML model back to about 79 false positives at 0.25 with unchanged true positives (−55 %).** On these frames that is better than the `.pt` (100), because the square letterbox that Core ML and the app use helps a little.

## Setup

- **Data.** The public test split of the [box2boxcam dataset](https://huggingface.co/datasets/Totorotibo84/box2boxcam-football-ball-detection-dataset): 396 frames, 219 balls, 149 of them small (≤ 16 px). One match, never used for training.
- **Models.** [v1.0](https://huggingface.co/Totorotibo84/box2boxcam-football-ball-detection) `.pt` and the published `.mlpackage`. The `.mlpackage` is an fp16 pipeline: an ML program, then Apple's NMS with confidence 0.1 and IoU 0.7, with classes padded to 80.
- **Scoring.** The repository's `scripts/evaluate_detector.py`, with matching at IoU ≥ 0.5, TP/FP per confidence threshold and a paired bootstrap over frames (2,000 resamples). It now accepts `.mlpackage` files and reports box-level agreement between two models (`--agreement`).
- **Core ML runs.** An Apple M5 Pro on macOS 26.7.1, with Python 3.12, torch 2.7.1, coremltools 9.0, numpy 2.3.5 and Ultralytics 8.4.153, the versions of the iOS repo's `scripts/fetch-models.sh`. On this Mac the Neural Engine compile of the model failed (`ANECCompile() FAILED`), so runs on `CPU_AND_NE` fell back to the CPU. An iPhone may behave differently.
- **PyTorch reconstructions.** These ran on Linux (CPU).

Cells below read TP/FP, with small TP in brackets, at confidence 0.25 / 0.35 / 0.45 / 0.55. Bootstrap differences are Core ML minus `.pt`.

## 1. First comparison: the published export against the `.pt`

| model | 0.25 | 0.35 | 0.45 | 0.55 | max TP at FP ≤ 25 / 60 / 100 / 200 |
|---|---|---|---|---|---|
| `.mlpackage` (app model) | 96/184 (54) | 91/98 (49) | 85/38 (46) | 76/18 (41) | 78 / 87 / 91 / 97 |
| `.pt` | 99/100 (55) | 92/52 (50) | 85/22 (46) | 82/4 (46) | 86 / 93 / 99 / 111 |

- **The `.pt` row reproduces the README exactly on the Mac.** A Linux CPU run gave 100/104 (56), a small platform difference.
- **The bootstrap difference is concentrated in false positives:**
  - FP@0.25: +84 [+50, +119]
  - FP@0.55: +14 [+6, +23]
  - TP@0.25: −3 [−8, +2], within noise
  - Max TP under a budget of 200 false positives: −14 [−24, −8]
- **Box agreement at 0.25 is moderate.** 162 boxes are matched, 118 come only from Core ML and 37 only from the `.pt`. 299 of 396 frames are identical. Mean IoU is 0.909 and the mean confidence difference is 0.034.

## 2. Ruling out precision and compute units

The script re-exports with the same settings as the published model: `nms=True`, `conf=0.1`, `iou=0.7`, imgsz 960. Ultralytics loads every Core ML model on `CPU_AND_NE` with no option to change it, so a small wrapper forced the compute units.

| run | model | units | 0.25 | 0.35 | 0.45 | 0.55 | FP@0.25 diff |
|---|---|---|---|---|---|---|---|
| R1 | published fp16 | CPU_AND_NE | 96/184 (54) | 91/98 (49) | 85/38 (46) | 76/18 (41) | +84 [+50, +119] |
| R2 | published fp16 | CPU_ONLY | 97/182 (55) | 92/98 (50) | 86/41 (47) | 77/19 (42) | +82 [+50, +116] |
| R3 | re-export fp16 | CPU_ONLY | 97/182 (55) | 92/98 (50) | 86/41 (47) | 77/19 (42) | +82 [+50, +116] |
| R4 | re-export fp32 | CPU_ONLY | 98/174 (56) | 92/93 (50) | 86/33 (47) | 76/17 (42) | +74 [+41, +108] |

- **The re-export reproduces the published model exactly.** R3 equals R2. Its spec is also identical: the same metadata, NMS settings and op precisions.
- **Compute units change nothing.** R1 and R2 are almost identical.
- **fp32 explains only about 8 of the extra false positives.** It does tighten boxes (mean IoU 0.973 against 0.891).

Note: with `nms=True` and no `quantize`, Ultralytics exports fp16 by default; fp32 needs `quantize=32`.

## 3. The export path rebuilt in PyTorch

The exporter's preparation was reproduced in PyTorch:
- The model is fused, with the `Detect` head in export mode.
- The `IOSDetectModel` wrapper pads classes to 80 and normalises boxes.
- Frames are letterboxed to 960×960 squares.
- A per-class NMS runs at confidence 0.1 and IoU 0.7 (torchvision).

| reconstruction | 0.25 | 0.35 | 0.45 | 0.55 |
|---|---|---|---|---|
| export graph + NMS IoU 0.7 (torchvision or Ultralytics) | 97/79 (55) | 92/39 | 87/12 | 77/4 |
| export graph, no NMS | 98/525 | 92/399 | 87/295 | 77/218 |
| export graph + NMS IoU 0.9 | 97/191 | 92/112 | 87/49 | 77/21 |
| `.pt` on square 960×960 frames | 97/79 (55) | 92/39 | 87/12 | 77/4 |

- **The design of the export is sound.** The extra boxes therefore appear inside Core ML.
- **The square letterbox is not a cause either.** It *lowers* the `.pt`'s false positives from 100 to 79. The `.pt` normally runs on a minimal rectangle of about 960×544.
- **The Core ML numbers sit close to a standard NMS at IoU 0.9.**

The two heads are not the explanation either. The `.pt` and the export both use the one-to-many head with NMS (`end2end False`).

## 4. Network or NMS stage?

The same network was exported without the NMS pipeline. Leaving `nms` unset keeps the one-to-many head, while `nms=False` would switch to the end2end head. Ultralytics then applies its own NMS in Python.

| run | model | 0.25 | 0.35 | 0.45 | 0.55 | FP@0.25 diff |
|---|---|---|---|---|---|---|
| R4 | fp32 + Apple NMS pipeline | 98/174 (56) | 92/93 (50) | 86/33 (47) | 76/17 (42) | +74 [+41, +108] |
| R5 | raw fp32, Ultralytics NMS | 98/77 (56) | 92/37 (50) | 86/10 (47) | 76/4 (42) | −23 [−36, −10] |
| R6 | raw fp16, Ultralytics NMS | 98/83 (56) | 92/36 (50) | 86/10 (47) | 77/4 (42) | −17 [−31, −4] |

A direct comparison of the raw fp32 Core ML outputs with the PyTorch export graph covered 60 frames and 1,134,000 anchors on CPU_ONLY:
- Score differences were 0 (mean and max).
- Box differences were at most 0.003 px.
- 122 anchors had a score ≥ 0.1 in both models.
- No anchor crossed 0.25.

**The converted network is exact. The extra false positives come from the NMS stage.**

## 5. How Apple's NMS reads `iouThreshold`

The published pipeline was called directly. Its `iouThreshold` and `confidenceThreshold` are optional inputs. The last column counts returned box pairs in the same frame that still overlap above a standard IoU of 0.7, which a standard NMS at 0.7 never returns.

| variant | CPU_ONLY | pairs | CPU_AND_NE (fell back) | pairs |
|---|---|---|---|---|
| V0 default inputs (as Ultralytics and the app call it) | 97/182, 92/98, 86/41, 77/19 | 364 | 96/184, 91/98, 85/38, 76/18 | 420 |
| V1 `iouThreshold` 0.7 passed explicitly | 97/182, 92/98, 86/41, 77/19 | 364 | 96/184, 91/98, 85/38, 76/18 | 420 |
| V2 `iouThreshold` 0.45 passed explicitly | 97/79, 92/34, 86/10, 77/4 | 0 | 96/78, 91/39, 85/11, 76/5 | 5 |
| V3 V0, then greedy NMS at standard IoU 0.7 | 97/85, 92/37, 86/10, 77/4 | 0 | 96/79, 91/39, 85/11, 76/5 | 0 |

- **The defaults are not the problem.** Passing 0.7 explicitly changes nothing.
- **Apple's NMS does not apply `iouThreshold` as a standard IoU threshold.** At 0.7 it keeps hundreds of pairs that overlap more than 0.7. At 0.45 it matches a standard NMS at 0.7 on these frames.
- **The Ultralytics exporter passes the user-facing `iou` (default 0.7) straight into Apple's NMS.** Its own `pipeline_coreml()` defaults to `iou=0.45`, but the exporter overrides that default.

## What it means for the app

- **The app sees the same duplicates.** `BallDetector.swift` runs the same pipeline through Vision (`VNCoreMLRequest`, `scaleFit`) without passing `iouThreshold`, so its candidate list very likely carries the same near-duplicate boxes. Most of them sit on top of a detection rather than elsewhere in the image, which may be why the camera behaves sensibly. But they inflate candidate counts, and they can split confidence between near-identical boxes.
- **The README's precision is reachable on the phone.** It describes the `.pt`, and with a working NMS the Core ML model matches or beats it on the test split.
- **This was measured through Core ML directly on a Mac, not through Vision on an iPhone.** Vision letterboxes the same way, but confirming on a device is still a next step.

## Proposed next steps

1. **Fix the duplicates in the app (recommended first).** Add a small greedy NMS at standard IoU 0.7 on Vision's results in `BallDetector.swift`, with the logic in `DockKitCore` and a check. It works with any model file and does not depend on Apple's NMS semantics. On the Mac it left 0 overlapping pairs on both compute units, and false positives fell from 184 to 79 at 0.25.
2. **Fix the model file.** Re-export v1.0 with `iou=0.45`, which gives the same benefit for any Core ML user of the published model. Alternatively, the app can pass `iouThreshold = 0.45` through `VNCoreMLModel.featureProvider` with the current file. Both rely on Apple's mapping, which left 5 residual pairs on the Neural Engine fallback.
3. **Report it to Ultralytics.** For detection exports with `nms=True`, the exporter should translate the standard IoU, or default to 0.45 as `pipeline_coreml()` does. This report, the probe script and the numbers above are the evidence.
4. **Publish Core ML numbers.** Report the shipped model's results next to the `.pt` in the README, measured with `evaluate_detector.py` on a `.mlpackage`, once a fix is in.
5. **Confirm on an iPhone.** Run the app's model latency benchmark (branch `feat/model-latency-benchmark` of the iOS fork) and a short logged match before and after the fix, to check candidate counts and that the extra NMS pass costs nothing measurable.

## Reproduce

- **Evaluation code** is on branch [`feat/core-ml-parity`](https://github.com/rileyw/box2boxcam-football-vision/tree/feat/core-ml-parity): `evaluate_detector.py` with `.mlpackage` support and `--agreement`, plus `box2boxcam.evaluation.agreement`.
- **One-command Mac scripts** are on branch [`ops/core-ml-parity-run`](https://github.com/rileyw/box2boxcam-football-vision/tree/ops/core-ml-parity-run). They need [uv](https://docs.astral.sh/uv/), and each opens a prefilled GitHub issue with its results:
  - `scripts/mac_parity_run.sh` (section 1)
  - `scripts/mac_fp32_experiment.sh` (section 2)
  - `scripts/mac_nms_experiment.sh` (section 4)
  - `scripts/mac_nms_probe.sh` (section 5)
  - helpers: `coreml_units_eval.py`, `coreml_raw_diff.py`, `coreml_nms_probe.py`

```sh
git clone -b ops/core-ml-parity-run https://github.com/rileyw/box2boxcam-football-vision.git
cd box2boxcam-football-vision
scripts/mac_parity_run.sh && scripts/mac_fp32_experiment.sh \
  && scripts/mac_nms_experiment.sh && scripts/mac_nms_probe.sh
```

## Caveats

- **The data is limited.** It covers one test match, one stadium and 396 frames, so the false-positive differences are large relative to their CIs, but the absolute numbers may not transfer to other venues.
- **No iPhone measurement yet.** On the Mac, the Neural Engine compile failed and Core ML fell back to the CPU.
- **The 0.45 ↔ 0.7 correspondence is empirical,** measured on these frames for this model. Apple does not document how its NMS applies `iouThreshold`. The app-side NMS avoids depending on it.
