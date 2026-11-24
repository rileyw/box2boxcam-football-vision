# Local training on an RTX 5070 Ti (fork-only)

Fork-only: local training and labelling tools around the box2boxcam v1.0 recipe, without the author's paid Hugging Face Jobs or private datasets. Nothing here is proposed upstream yet.

| File | What |
|---|---|
| `MODEL_IMPROVEMENT_PLAN.md` | Hardware findings, machine roles, ranked experiments and their evaluation gate. |
| `train_local.py` | The v1.0 recipe (COCO YOLO26s, sports-ball transfer, 960 px) on the public dataset, or on `--data <yaml>`. About 12 min per seed. |
| `smoke_train.py` | Throughput and memory smoke test. |
| `RESULTS.md` | Results, plus the log of every look at the test split. |
| `DATA_GUIDE.md` | Producing more training data and media: consent, what to film, filming with the app, labelling, demo clips. |
| `analysis/` | Error analysis on val (`error_analysis.py`): kinds of false positives and misses. |
| `teacher/` | Pseudo-labelling pipeline: ensemble teacher, calibration gate, Label Studio review, dataset builder. See `teacher/README.md`. |
| `eval/*.json` | `evaluate_detector.py` reports for the public baseline against v1.0. |

Setup (not committed): a `.venv` with torch 2.7.1+cu128, ultralytics 8.4.153, numpy ≤ 2.3.5 and this repo installed with `-e`; the public dataset under `data/public/`. Paths in the scripts are relative to this folder. Frames of filmed matches (`pseudo/`) show people: they are git-ignored and must stay local.
