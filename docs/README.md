# Documentation index

| Topic | Page |
|---|---|
| Results: box2boxcam v1.0 (released model) | [mobile_auto_field_v006_20261111.md](mobile_auto_field_v006_20261111.md) |
| Fine-tuning vs stock YOLO, training curves | [README, Results](../README.md#fine-tuning-vs-stock-yolo) (figures in [assets](assets/)) |
| Intermediate experiments | [+ automatic field labels, first batch](mobile_auto_field_v005_20261111.md), [first field model](mobile_field_v004_20261014.md) |
| Model size comparison (Nano / Small, 960 / 1280) | [mobile_size_comparison_20261015.md](mobile_size_comparison_20261015.md) |
| Earlier experiments | [personal test](mobile_personal_test_20261010.md), [TV comparison](mobile_tv_comparison_20261010.md), [sports-ball transfer](mobile_sports_ball_transfer_20261010.md) |
| What did not work | [negative_results.md](negative_results.md) |
| Core ML parity of the app model (export vs `.pt`, NMS finding) | [core_ml_parity_20261123.md](core_ml_parity_20261123.md) |
| Camera-guidance benchmark (SoccerTrack v2) | [camera_guidance_benchmark.md](camera_guidance_benchmark.md) |
| Dataset audit | [dataset_assessment.md](dataset_assessment.md) |
| Reserved-match annotation protocol | [field_validation_20261021.md](field_validation_20261021.md) |
| Research survey (vision and framing), exploratory | [vision_etat_de_lart_20261105.md](vision_etat_de_lart_20261105.md) |
| Hugging Face dataset / model cards | [hf_dataset_README.md](hf_dataset_README.md), [hf_highres_README.md](hf_highres_README.md), [hf_model_README.md](hf_model_README.md) |

## Vocabulary

Public names are used in the prose. File names, config keys and dataset folders keep internal ids, which this table translates.

| Public name | Internal id (files, configs, folders) |
|---|---|
| stock YOLO26s | `yolo26s.pt`, COCO weights, class 32 "sports ball" only |
| first field model | `mobile_v004` dataset, `*_field_*` configs (Nano and Small sizes) |
| + automatic field labels, first batch | `mobile_v005`, `configs/mobile_small_auto_field_960.json` |
| box2boxcam v1.0 (released model) | `mobile_v006`, `configs/mobile_small_auto_field_v006_960.json` |
| multimodal proposer | a third-party vision-language model API used to propose ball boxes before human review (config and script names contain `luna`) |
| reserved match | the 14 October 2026 iPhone session, never used for training |
