# Teacher pseudo-labelling pipeline (local, RTX 5070 Ti)

Turns a new match video into training labels. A large teacher (YOLO26x at 960 px) proposes balls. Confident frames are accepted automatically; uncertain ones go to a human in Label Studio. It follows the author's rules (`docs/dataset_assessment.md`):
- the final test stays human-labelled
- teacher and student share train, never test
- a human decides ambiguous cases

```
video ─▶ pseudo_label.py ─▶ auto labels ──────────────────────┐
                         └▶ review/tasks.json ─▶ Label Studio ─▶ import_review.py ─▶ build_dataset.py ─▶ train_local.py --data
```

All commands run from `~/workspace/football-vision/training` with `.venv/bin/python -I`.

## 0. Teacher (once)

The teacher is an **ensemble**: the two public baseline students (seeds 42 and 7), each run on the frame and on its mirror image, with their boxes merged by weighted box fusion (`teacher/ensemble.py`). That's 4 members, and a box only one member sees gets a quarter of its score. Flip-only TTA is deliberate: Ultralytics' built-in TTA also shrinks the image, which hurts balls a few pixels wide.

```sh
.venv/bin/python -I teacher/calibrate.py --teacher eval/models/base_s42.pt eval/models/base_s7.pt --flip \
    --student eval/models/base_s42.pt --out teacher/thresholds.json
```

`calibrate.py` runs on **val only** and refuses the test split. It does two things:
- **Gate:** the teacher must find at least as many balls as the student at every false-positive budget, and more small balls at 0.25.
- **Thresholds:** `high` is the lowest confidence where the teacher's val precision is at least 95 % (auto-accept). `low` (0.15) is the review floor.

The teacher setup is recorded in `thresholds.json`, and `pseudo_label.py` refuses a different one.

Candidates tried on 2026-11-23 (val; "most balls at FP ≤ 25/60/100/200", then small balls at 0.25; student `base_s42` = 176/203/209/212, 82):

| teacher | val result | gate |
|---|---|---|
| YOLO26x, v1.0 recipe, batch 4 | 156/195/202/206, 80 | FAIL (plateaued below the student) |
| student + flip | 186/205/215/216, 83 | pass |
| 2 seeds, no flip | 182/201/206/213, 84 | FAIL |
| **2 seeds + flip (current)** | **193/207/213/217, 85**; auto-accept at 0.70 | **pass** |
| v1.0 + 2 seeds + flip | 199/212/217/220, 86 | pass, but v1.0 may have trained on the val clip, so it may be optimistic |

All of these are on val, which comes from the same match as train, so the gains are modest and optimistic. Use `--audit-auto` (below) to measure the real precision of auto-accept on new footage.

## 1. Pseudo-label a match

One video per match, with a role:

```sh
# a match for training
.venv/bin/python -I teacher/pseudo_label.py --video ~/videos/club_20261130.mp4 --match club_20261130 \
    --role train --teacher eval/models/base_s42.pt eval/models/base_s7.pt --flip \
    --thresholds teacher/thresholds.json --stride 15 --review-empty 0.05 --audit-auto 0.1
# a match kept as the next held-out test: every frame goes to review, none to training
.venv/bin/python -I teacher/pseudo_label.py --video ~/videos/club_20261207.mp4 --match club_20261207 \
    --role reserved --teacher eval/models/base_s42.pt eval/models/base_s7.pt --flip \
    --thresholds teacher/thresholds.json --stride 30
```

Every sampled frame (`--stride 15` is 2 per second at 30 fps) gets one status:
- **auto** (train only): every teacher box ≥ `low` is also ≥ `high`. Its boxes are written to `pseudo/<match>/labels/` without review.
- **review**: some box is between `low` and `high`, or the match is reserved. The frame goes to `pseudo/<match>/review/tasks.json` with the teacher's boxes as pre-annotations.
- **skipped**: no box ≥ `low`. The frame is not used, because a missed small ball would otherwise teach "no ball here". `--review-empty 0.05` sends 5 % of these to review as negative candidates.
- `--audit-auto 0.1` sends 10 % of the would-be auto frames to review instead, marked "audit of an auto-accepted frame". Comparing the teacher's boxes with your corrections on those frames shows how precise auto-accept really is on this footage. If it falls well below 95 %, raise `high` in `thresholds.json`.

Check `pseudo/<match>/summary.json` for the counts. The frames show people: keep them local, and don't publish them without consent and face blurring.

## 2. Review in Label Studio

```sh
teacher/label_studio.sh      # installs Label Studio into .venv-ls on first run, then starts it
```

The script pins `sentry-sdk<3`, because Label Studio 1.23.2 crashes at start with sentry-sdk 3. It serves frames from `pseudo/`, keeps its data in `.ls-data/`, and turns off update checks and error reporting.

Open http://localhost:8080 in the Windows browser (WSL forwards localhost). The first visit asks you to sign up; that's a local account only. Then, once per match:
1. **Create a project.** Under Settings → Labeling Interface → Code, paste the author's `configs/label_studio_all_balls.xml`. It draws one box per visible ball, has an "Exclure image" choice, and shows the teacher's confidences.
2. **Add the match's storage.** Settings → Cloud Storage → Add Source Storage → Local files. Absolute path: `<training>/pseudo/<match>`. It must be the match subfolder: Label Studio refuses the serving root itself. Don't sync; the tasks already point at the files.
3. **Import** `pseudo/<match>/review/tasks.json`. The teacher's boxes appear as predictions.
4. **Review each frame:** fix or remove wrong boxes, add missed balls, then **Submit**.
   - Submitting with no box is an explicit "no ball" label.
   - "Exclure image" drops the frame.
   - **Skip** leaves the frame unused.
5. **Export:** Export → JSON.

Only submitted annotations count. Unreviewed and skipped tasks are simply not used.

```sh
.venv/bin/python -I teacher/import_review.py --match club_20261130 --export ~/Downloads/project-export.json
```

## 3. Build the dataset and train

```sh
.venv/bin/python -I teacher/build_dataset.py --name plus_club --matches club_20261130 club_20261207
.venv/bin/python -I train_local.py --data data/plus_club.yaml --seed 42 --name plus_club_s42
.venv/bin/python -I train_local.py --data data/plus_club.yaml --seed 7  --name plus_club_s7
```

`build_dataset.py` assembles the data from symlinks:
- **Train:** public train, plus each train-role match's auto-accepted and human-reviewed frames (reviewed labels win).
- **Val:** stays public val, so checkpoint selection matches the baseline.
- **Reserved matches** never enter train. Their reviewed frames become `data/reserved/<match>/` and must all be human-reviewed.

Then gate against the baseline as in `MODEL_IMPROVEMENT_PLAN.md`:
- a paired bootstrap at equal false-positive budgets, 2 seeds agreeing
- on the original test split **and** the new reserved match
- logged in `RESULTS.md`

## Tested (2026-11-23, smoke test)

The baseline student acted as the teacher, on a 12 s stand-in video stitched from public val frames:
- **calibrate:** ran and gated, and refused the test split.
- **pseudo_label:** both roles; all 14 auto-accepted boxes matched the human ground truth at IoU ≥ 0.5 (median IoU 0.85).
- **import_review:** a simulated export with an excluded frame and an empty frame.
- **build_dataset:** reserved frames stayed out of train.
- **train_local --data:** a 1-epoch run read the assembled dataset.

**Label Studio round trip (2026-11-23)**, Label Studio 1.23.2, driven through its REST API:
- 33 tasks imported with their teacher predictions.
- Frames served once the per-match storage was added: 200, a 1920×1080 JPEG.
- Six simulated reviewer decisions, all imported correctly from the real export: accept, moved box, added ball, "Exclure image", empty submission, and a skip that was ignored.
- **Not exercised:** the browser UI. The reviews were submitted through the API, which stores the same annotation records the UI does.
