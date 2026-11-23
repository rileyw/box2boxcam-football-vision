# Improving the box2boxcam ball model: where to start

Written 2026-11-23. It covers the two machines: this WSL2 box with an RTX 5070 Ti, and a MacBook Pro M5 Pro. Local work lives in `training/`, which is untracked; nothing here is pushed.

## 1. What was verified on this machine

| Check | Result |
|---|---|
| GPU visible in WSL2 | Yes: RTX 5070 Ti, 16 GB (about 2.7 GB used by the Windows desktop), driver 620.73 / CUDA UMD 13.5 |
| Host | 8 cores, 31 GB RAM in WSL, data on ext4 under `/home` (keep it off `/mnt/c`) |
| PyTorch | `torch 2.7.1+cu128` from the cu128 index has `sm_120` kernels, so it runs. This is the same torch version as the Core ML export environment. Matmul throughput is about 110 fp16 TFLOPS. |
| Stack | `ultralytics==8.4.153` (the author pins and checks it), `numpy 2.3.5`, `opencv-python-headless`, project installed `--no-deps -e` |
| v1.0 recipe, end to end | Runs: COCO YOLO26s, `SportsBallTransferTrainer`, 960 px, AdamW, mosaic 0, scale 0.3, AMP checks pass |
| Throughput (YOLO26s, 960 px, 1 epoch = 1,673 images, including startup and validation) | batch 8: 46 s, 5.0 GiB peak; **batch 16: 41 s, 9.7 GiB** (fastest); batch 32: overflows into shared memory under WSL and crawls, so don't use it |
| Full 30-epoch baseline | Not run. Estimated about 20–25 minutes per seed at batch 8, against the author's 25 minutes on an A10G with 2,806 images. |

Not verified:
- MPS training on the Mac (not run).
- Neural Engine latency on an iPhone.
- Training with a P2 head.

## 2. Machine roles

- **RTX 5070 Ti (WSL), all training and PyTorch evaluation.** This includes seed replicates, sweeps, teacher and pseudo-label inference, and `evaluate_detector.py` on `.pt` models. About 2.5 full v1.0-size runs fit in an hour.
- **M5 Pro, everything Core ML.**
  - Export with the pinned `fetch-models.sh` versions and the NMS fix.
  - Core ML parity and false positives with `evaluate_detector.py --agreement` and the probe scripts already on `ops/core-ml-parity-run`.
  - Xcode builds, and installing the app on an iPhone to run the in-app latency benchmark (`feat/model-latency-benchmark`).
- **Not MPS training.** The Mac could train, but slower than the 5070 Ti, and Ultralytics on MPS disables AMP and has historic op and numeric issues, so results wouldn't be comparable to the CUDA runs. It's not worth debugging when the 5070 Ti is idle. (Not tested.)
- **Neural Engine.** The ANE compile of the ball model failed on this Mac in every parity run, so **Neural Engine latency has to be measured on an iPhone**, not the Mac.

The bridge between the machines: the Mac has no copy and paste, so candidate weights have to reach it without typing tokens. The simplest route is a GitHub release asset on the `rileyw` fork, which the Mac fetches with `curl`, no authentication needed. The weights are AGPL either way. Alternatively, use a private Hugging Face repo after a one-time `hf auth login` on the Mac.

## 3. What can be reproduced from public data

- **v1.0 trained on about 3,283 private images.** That is `mobile_v004`, 2,806 images including third-party and YouTube frames, plus 477 auto-judged images. It was validated on a private 168-image set, and its configs point to `*-private` Hugging Face repos.
- **The public dataset** (`Totorotibo84/box2boxcam-football-ball-detection-dataset`, CC BY 4.0) contains:
  - train: 1,673 images of the author's own footage
  - val: 292 images, **another clip of the same match as train**, so optimistic
  - **test: 396 images of the held-out match, the only independent split**
  - `third_party/annotations.jsonl`: 1,723 frames you'd have to rebuild by downloading 7 YouTube/TikTok videos (rights stay with the owners)
- **An exact v1.0 reproduction is not possible.** A faithful *public-data baseline* is: same recipe, public train/val. Rebuilding the third-party frames would bring the data close to v1.0's, but whether to download those videos is your call.

Baseline commands (already set up in `training/`):

```sh
cd ~/workspace/football-vision/training
# one-time (done): .venv with torch 2.7.1+cu128, ultralytics 8.4.153, numpy<=2.3.5, project -e
.venv/bin/python -I train_local.py --seed 42 --name base_s42   # about 20-25 min
.venv/bin/python -I train_local.py --seed 7  --name base_s7
# score on the held-out match against v1.0 (paired bootstrap)
cd ../forks/box2boxcam-football-vision
../../training/.venv/bin/python -I scripts/evaluate_detector.py \
  --weights ../../training/runs/base_s42/weights/best.pt --compare <v1.0 .pt> \
  --frames-dir ../../training/data/public/images/test --labels ../../training/data/public/labels/test \
  --device 0 --agreement
```

`train_local.py` is the v1.0 recipe as in `scripts/train_mobile_hf.py`: COCO yolo26s, full sports-ball transfer, 960 px, batch 8, 30 epochs, patience 8, AdamW lr 0.001, cosine, mosaic 0, scale 0.3, AMP, deterministic. Checkpoint selection uses **val only**, as the author does. Expect the public-only baseline to land *below* v1.0, since it has about half the data. It is the reference for the experiments below, not v1.0.

## 4. The key diagnostic: what limits recall

v1.0 inference only, at three input sizes, on **val** (the test split is left untouched):

| imgsz | 0.25 TP/FP (small) | 0.55 TP/FP (small) | max TP at FP ≤ 25 / 60 / 100 / 200 |
|---|---|---|---|
| 960 | 207/45 (83) | 170/20 (62) | 188 / 210 / 215 / 217 |
| 1280 | 207/64 (88) | 194/33 (82) | 178 / 207 / 212 / 218 |
| 1536 | 206/96 (89) | 193/42 (85) | 186 / 201 / 206 / 216 |

- **Upscaling at inference finds no new balls.** TP@0.25 is flat at 207. It raises the confidence of balls already found and adds false positives at the same rate, so it's no better at equal false-positive budgets. This agrees with the author's Nano-1280 result.
- **Small-ball recall is 69 % on val (83/120) but 37 % on the held-out match (55/149).** The dominant gap is **generalisation to a new match** (venue, light, night, camera position), not pixels alone.

So more independent match footage is the highest-value lever. Architecture changes for small objects are worth testing only *after* a public baseline exists and only if they move the held-out match, not val.

## 5. First experiments, ranked by expected value per 5070 Ti GPU-hour

Evaluation gate for every candidate:
- Select checkpoints on val only.
- Then a paired bootstrap on the held-out test split against the public baseline, with **≥ 2 seeds (42, 7), both in the same direction**.
- Compare at equal false-positive budgets (25/60/100/200) as well as at 0.25.
- **Core ML parity of the exported candidate on the Mac, with the NMS fix** (Apple NMS `iou=0.45`, or app-side NMS 0.7). The shipped model is what counts, and a pipeline export at `iou=0.7` inflates false positives about 1.8×.
- Every look at the test split erodes it. Keep a log of how many times it has been used, and plan a **new reserved match**.

| # | Experiment | GPU time | Why / what must be true |
|---|---|---|---|
| 0 | **Ship the NMS fix** (re-export `iou=0.45` or app-side NMS 0.7) | 0 | Already measured: −55 % false positives at equal TP. Every later number assumes it. |
| 1 | **Public baseline, 2 seeds** | ~1 h | Needed as the reference, and it measures seed noise (the author saw ±60 FP between seeds). |
| 2 | **New footage: film 2–3 matches** (different venues, day and night), label with the author's proposer and click-review workflow, hold one out as a fresh reserved test, train with the rest | ~1 h per retrain plus labelling time | Highest expected gain given the 69 % vs 37 % gap. Requires players' consent, blurring and labelling time. Worth it if a rerun shows held-out small recall rising. |
| 3 | **Big teacher → pseudo-labels on unlabelled frames** (YOLO26l/x at 1280 on the 5070 Ti, high-confidence plus human review of unsure ones, as the author did with "auto-judged") | 2–4 h teacher + 1 h student | Cheaper labels for #2's footage. Requires unlabelled footage; the teacher must beat the student on val *small* recall. |
| 4 | **P2 (stride-4) head**, `yolo26s-p2.yaml` + COCO backbone | ~1.5 h (2 seeds; about 1.5× slower per epoch, estimated) | The model card's "80 % of misses have no candidate" fits a stride-8 limit, but the inference-resolution test above did not recover them. Worth it only if it beats the baseline on held-out small recall at equal FP. Caveats: the transfer trainer can't keep the COCO sports-ball head, so its prior is lost, and the phone cost (P2 at 960 adds compute) must pass the iPhone latency benchmark. |
| 5 | **Train at 1280 or with crop/tiling augmentation**, infer at 960 or tiles | ~2 h | Lower priority: the author's 1280 retrain didn't help, and on-phone tiling multiplies latency. |
| 6 | **Hard negatives** (spare balls, cones, lights) | ~1 h | After #0 the remaining false positives are smaller. Do it if the residual false positives are dominated by static off-pitch objects. The camera policy's static filter already targets those. |

Don't repeat these (documented in `docs/negative_results.md`):
- WASB weights
- 9-channel frame stacking
- small-ball copy-paste
- gated low-threshold re-detection
- single-seed scale-augmentation "gains"

## 6. First two weeks

**Days 1–2 (Linux):**
1. Run #1 (two baseline seeds) and score them on test once against v1.0.
2. Write down seed noise (the bootstrap CIs).
3. Add `train_local.py` and the commands to a branch only if they prove useful upstream. A local runner could be a small PR, since the author's runner needs paid Hugging Face Jobs.

**Days 2–3 (Mac):** ship #0. On the iOS fork, build the app-side NMS (`fix/ball-candidate-dedup`) and confirm on an iPhone with the latency benchmark.

**Week 1, film:** shoot 1–2 matches with the app (that also tests the app, the author's stated bottleneck). Extract frames on Linux and label (#3 teacher proposals plus review). Keep one match fully reserved.

**Week 2:**
1. Retrain the baseline with the new footage, 2 seeds.
2. Gate on the *new* reserved match plus the original test.
3. Export the winner on the Mac (sketch below) and check parity and iPhone latency.
4. If the held-out small recall gap persists, run #4 (P2) under the same gate.

## 7. Mac one-command step (sketch, not yet written)

`scripts/mac_candidate_eval.sh <release-tag>` on the `ops/core-ml-parity-run` branch, in the style of `mac_parity_run.sh`:
1. `curl -L` the candidate `.pt` from `https://github.com/rileyw/box2boxcam-football-vision/releases/download/<tag>/candidate.pt` into `parity-run/candidates/<tag>/`.
2. Export to Core ML with `nms=True, conf=0.1, iou=0.45, quantize=16, imgsz=960` (published settings plus the NMS fix).
3. Run `evaluate_detector.py --weights candidate.mlpackage --compare candidate.pt --agreement`, then the same against the v1.0 `.mlpackage` (published with the same fix applied via `coreml_nms_probe.py` V2).
4. Open a prefilled GitHub issue with both tables, using the same URL trick as the existing scripts.

## 8. Licensing and plumbing

- **AGPL-3.0 everywhere.** That covers Ultralytics, the code and the v1.0 weights, so any model trained with Ultralytics or from v1.0 is AGPL, and publishing weights or a hosted service means publishing source. A closed or commercial path would need a different detector (for example RF-DETR, Apache-2.0) and separate data provenance.
- **Private repositories.** Configs `configs/mobile_*.json` name the author's private Hugging Face repos, and `scripts/train_mobile_hf.py` expects HF Jobs and a dataset tarball with SHA-256 manifests. `train_local.py` avoids all of that.
- **The test split is the reserved match.** Never train on it or select on it. The public `val` shares its match with train.
- **Third-party frames.** Rebuilding them means downloading YouTube/TikTok videos whose rights stay with their owners. The decision is yours, and those frames must never be redistributed.
- **Footage privacy.** New footage needs players' consent and head blurring before any publication. The author's pipeline blurs heads with a pose model.
