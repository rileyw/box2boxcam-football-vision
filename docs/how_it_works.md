# How the box2boxcam ball model works, and how it is trained

This explains the system as it stands in the `rileyw` forks on 23 November 2026 (a fork-only note, not part of the upstream docs): the model, how the iPhone app uses it, how it is trained, and the local training loop added around it. Numbers come from the author's docs and from our own runs, cited where they appear.

## 1. The big picture

An iPhone on a DockKit gimbal films the match. Three models run on the phone about 30 times a second, a camera policy decides where to look, and the phone turns the gimbal's motors.

```mermaid
flowchart LR
    CAM["iPhone camera<br/>1920×1080, 30 fps"] --> BALL["Ball detector<br/>box2boxcam v1.0<br/>YOLO26s @ 960"]
    CAM --> PEOPLE["Person detector<br/>stock YOLO26s @ 960"]
    PEOPLE --> POSE["Pose model<br/>YOLO26s-pose @ 384<br/>on person crops"]
    BALL --> POLICY["Hybrid camera policy<br/>follow the ball while its track is reliable,<br/>else the densest group of players"]
    PEOPLE --> POLICY
    POSE --> POLICY
    POLICY --> MOTOR["DockKit velocity commands<br/>pan / tilt, ≤ 0.30 rad/s"]
    MOTOR --> GIMBAL["Gimbal motors"]
    GIMBAL -. "camera moves" .-> CAM
```

Only the **ball detector** is trained by this project. The person and pose models are stock Ultralytics models exported to Core ML (`scripts/fetch-models.sh` in the iOS repo).

## 2. The ball detector

The ball detector is a **YOLO26s** (about 9.5 M parameters), a single-stage detector. It looks at one image and predicts, for thousands of positions at once, "is there a ball here, and where exactly is its box?".

```mermaid
flowchart TB
    IN["Frame 1920×1080"] --> LB["Letterbox to 960×960<br/>scale to fit, pad with grey"]
    LB --> BB["Backbone<br/>convolutions + attention block<br/>extracts features"]
    BB --> NECK["Neck<br/>mixes coarse and fine features"]
    NECK --> P3["Stride 8 grid<br/>120×120 = 14,400 cells<br/>small objects"]
    NECK --> P4["Stride 16 grid<br/>60×60 = 3,600 cells"]
    NECK --> P5["Stride 32 grid<br/>30×30 = 900 cells<br/>large objects"]
    P3 --> HEAD["Detection head<br/>per cell: 4 box values + 1 ball score"]
    P4 --> HEAD
    P5 --> HEAD
    HEAD --> RAW["18,900 candidate boxes<br/>raw output shape (1, 5, 18900)"]
    RAW --> THR["Keep score ≥ threshold"]
    THR --> NMS["Non-maximum suppression<br/>drop boxes overlapping a better one, IoU > 0.7"]
    NMS --> OUT["Ball boxes + confidences"]
```

Key points:
- **Small balls are the hard part.** At 960 px a far ball is often under 16 px wide, only about 2 cells of the finest (stride 8) grid. The author reports that about 80 % of misses produce no candidate at all (`docs/negative_results.md`). Our test of running v1.0 at larger input sizes (up to 1536 px) found no extra balls, which suggests the limit is generalising to new matches rather than resolution (`MODEL_IMPROVEMENT_PLAN.md` §4).
- **Two heads exist, one is used.** YOLO26 has an NMS-free one-to-one head ("end2end") and a one-to-many head that needs NMS. Both v1.0 and its Core ML export use the **one-to-many head plus NMS** (`end2end: False`).

### Sports-ball transfer: how training starts

Training does not start from scratch. COCO-pretrained YOLO26s already detects 80 classes, one of them "sports ball" (class 32). The author's `SportsBallTransferTrainer` keeps the whole network and **only the classifier row for sports ball**, so training starts from a model that already knows roughly what a ball is.

```mermaid
flowchart LR
    COCO["COCO YOLO26s<br/>80-class classifier rows"] --> KEEP["Keep all backbone, neck and box weights<br/>keep classifier row 32 'sports ball'<br/>in both head branches"]
    KEEP --> ONE["1-class model<br/>'sports ball' = row 0<br/>same prediction as COCO's ball class"]
    ONE --> FT["Fine-tune on football frames"]
```

## 3. How the app uses the model, and the NMS problem we found

The phone runs a **Core ML** export of the model. The published export is a pipeline: the network in fp16, then Apple's own NMS stage with confidence 0.1 and IoU 0.7, with classes padded to 80 (an Ultralytics workaround).

```mermaid
sequenceDiagram
    participant Cam as Camera frame
    participant V as Vision (scaleFit 960×960)
    participant N as Core ML network (fp16)
    participant A as Apple NMS stage (iou 0.7, conf 0.1)
    participant D as CandidateSuppression (new, IoU 0.7)
    participant T as Static filter, trackers, hybrid policy
    Cam->>V: pixel buffer + orientation
    V->>N: letterboxed 960×960 image
    N->>A: 18,900 scores + boxes
    A->>V: boxes after Apple NMS
    Note over A: Reads 0.7 like a standard NMS at about 0.9,<br/>so it keeps near-duplicate boxes
    V->>D: candidates (score = object × label confidence)
    D->>T: duplicates removed (best box always kept)
```

What the Core ML parity research found ([`docs/core_ml_parity_20261123.md`](https://github.com/rileyw/box2boxcam-football-vision/blob/feat/core-ml-parity/docs/core_ml_parity_20261123.md) on branch `feat/core-ml-parity`):

```mermaid
flowchart TD
    Q["Core ML export: 96 TP / 184 FP at 0.25<br/>vs .pt 99 / 100"] --> P{"fp16 precision?"}
    P -- "fp32 export: 98 / 174" --> NOT1["Not the main cause"]
    Q --> U{"Neural Engine vs CPU?"}
    U -- "same numbers" --> NOT2["Not the cause"]
    Q --> G{"Network conversion?"}
    G -- "raw outputs identical to PyTorch" --> NOT3["Not the cause"]
    Q --> S{"Apple NMS stage?"}
    S -- "same network + standard NMS: 98 / 77" --> CAUSE["Cause: Apple NMS reads iou 0.7<br/>like a standard NMS at ~0.9"]
    CAUSE --> FIX1["Fix in the app: CandidateSuppression<br/>branch fix/ball-candidate-dedup"]
    CAUSE --> FIX2["Or re-export with iou=0.45"]
```

## 4. Training the model

### Data and splits

```mermaid
flowchart LR
    subgraph PUB["Public dataset (CC BY 4.0, faces blurred)"]
        TR["train<br/>1,673 frames<br/>author's matches"]
        VA["val<br/>292 frames<br/>same match as train"]
        TE["test<br/>396 frames, 219 balls<br/>held-out match"]
    end
    subgraph PRIV["Private (author only)"]
        V1D["v1.0 training set<br/>~3,283 frames incl. third-party<br/>and auto-judged labels"]
    end
    TR --> TRAIN["Training"]
    V1D -. "v1.0 only" .-> TRAIN
    VA --> SEL["Checkpoint selection<br/>(best epoch)"]
    TE --> GATE["Final comparison only<br/>never used to choose anything"]
```

- **val is optimistic:** it is another clip of the training match, so it favours whatever fits that match.
- **test is the only independent judge.** Every look at it is logged in `training/RESULTS.md`, and the plan calls for new reserved matches because each look wears it out.

### The recipe (v1.0, reproduced locally in `training/train_local.py`)

```mermaid
flowchart TB
    START["COCO YOLO26s"] --> TRANSFER["Sports-ball transfer"]
    TRANSFER --> LOOP{"Epoch 1..30"}
    LOOP --> AUG["Load batch of 8 frames at 960 px<br/>augment: scale ±30 %, flips, colour<br/>no mosaic, no mixup"]
    AUG --> FWD["Forward pass → 18,900 predictions per frame"]
    FWD --> LOSS["Loss: box regression + ball score<br/>against the labelled boxes"]
    LOSS --> OPT["AdamW, lr 0.001, cosine decay<br/>mixed precision"]
    OPT --> VAL["Validate on val"]
    VAL --> BEST{"Better than best so far?"}
    BEST -- yes --> SAVE["Save best.pt"]
    BEST -- no --> PAT{"8 epochs without gain?"}
    SAVE --> LOOP
    PAT -- no --> LOOP
    PAT -- yes --> STOP["Stop early"]
    LOOP -- done --> STOP
```

On the RTX 5070 Ti this takes **about 12 minutes per seed** (30 epochs, 5.3 GiB). The author trains the same recipe on paid Hugging Face Jobs (A10G).

### What the public baseline showed (`training/RESULTS.md`)

Trained on public data only (about half of v1.0's data), seeds 42 and 7, scored on the held-out test match:
- **At equal false-positive budgets it matches v1.0**: +1 to +8 TP, every CI includes zero.
- **At a fixed threshold it is more confident**: more balls found, including +10 to +14 small balls at 0.25, but also more false positives.
- **Seeds disagree a lot at fixed thresholds** (19 FP apart at 0.45) but stay within about ±5 TP at equal budgets.

So experiments are judged at equal false-positive budgets, with at least 2 seeds.

## 5. Improving the model: the local loop

New footage is the main lever. The pipeline in `training/teacher/` turns a filmed match into training labels with a large **teacher** model and a human reviewer.

```mermaid
flowchart TB
    FILM["Film a match<br/>(players' consent)"] --> ROLE{"Role of this match?"}
    ROLE -- "train" --> PL["pseudo_label.py<br/>2 frames/s → teacher YOLO26x @ 960"]
    ROLE -- "reserved (new test)" --> PLR["pseudo_label.py --role reserved<br/>every frame to review"]
    PL --> TRI{"Teacher boxes per frame"}
    TRI -- "all ≥ high (95 % precise on val)" --> AUTO["Auto-accepted labels"]
    TRI -- "some between low and high" --> REV["Review queue"]
    TRI -- "none ≥ low" --> SKIP["Skipped<br/>(avoids false 'no ball' labels)"]
    PLR --> REV
    REV --> LS["Label Studio<br/>fix, add, exclude, submit"]
    LS --> IMP["import_review.py<br/>human labels only"]
    AUTO --> BUILD["build_dataset.py<br/>public train + match frames<br/>val stays public val"]
    IMP --> BUILD
    IMP -- "reserved match" --> RES["data/reserved/&lt;match&gt;<br/>new held-out test, never trained on"]
    BUILD --> TRAIN2["train_local.py --data …<br/>seeds 42 and 7"]
```

The teacher itself is trained once with the same recipe (YOLO26x, batch 4 to fit 16 GB). It is used only if `calibrate.py` shows it beats the student **on val**, with more balls at every false-positive budget and more small balls. That script also sets the auto-accept threshold, and it refuses to run on the test split.

## 6. Deciding whether a new model is better

```mermaid
flowchart TD
    CAND["Candidate model<br/>(2 seeds)"] --> T1["Score on the original test match<br/>and every reserved match"]
    BASE["Baseline<br/>(same 2 seeds)"] --> T1
    T1 --> BOOT["Paired bootstrap over frames<br/>2,000 resamples"]
    BOOT --> C1{"More balls at equal FP budgets<br/>25 / 60 / 100 / 200,<br/>both seeds agreeing?"}
    C1 -- no --> REJ["Reject, record in negative results"]
    C1 -- yes --> EXP["Export to Core ML on the Mac"]
    EXP --> PAR{"Core ML parity OK<br/>(with the NMS fix)?"}
    PAR -- no --> FIXEXP["Fix the export"]
    PAR -- yes --> LAT{"iPhone latency OK<br/>(app keeps ~30 Hz)?"}
    LAT -- no --> REJ
    LAT -- yes --> SHIP["Candidate for the app"]
```

The scoring is `scripts/evaluate_detector.py`:
- A box counts as found at IoU ≥ 0.5.
- Small balls (≤ 16 px) are counted separately.
- "Max TP at FP ≤ budget" compares models at the same false-positive cost, which removes differences in confidence calibration.

## 7. Who does what

```mermaid
flowchart LR
    subgraph LINUX["Linux / WSL2, RTX 5070 Ti 16 GB"]
        L1["Training: students and teacher"]
        L2["PyTorch evaluation"]
        L3["Pseudo-labelling"]
        L4["Label Studio review"]
    end
    subgraph MAC["MacBook Pro M5 Pro"]
        M1["Core ML export + parity"]
        M2["Xcode builds"]
        M3["iPhone install + latency benchmark"]
    end
    subgraph PHONE["iPhone + gimbal"]
        P1["Matches: footage + logs"]
    end
    L1 -- "weights" --> M1
    M2 --> P1
    P1 -- "video" --> L3
```

## Where things live

| What | Where |
|---|---|
| Training, eval and research code | fork `rileyw/box2boxcam-football-vision` (`feat/core-ml-parity`, `ops/core-ml-parity-run`) |
| iOS app changes | fork `rileyw/box2boxcam-ios` (`fix/ball-candidate-dedup`, `feat/model-latency-benchmark`, …) |
| Local training, teacher pipeline, results | the local `training/` workspace, not in this repository: `train_local.py`, `teacher/`, `RESULTS.md` |
| Plans and research | the local workspace, not in this repository: `MODEL_IMPROVEMENT_PLAN.md`, `RESEARCH.md`, `drafts/` |
