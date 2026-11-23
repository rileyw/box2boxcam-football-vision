import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from box2boxcam.evaluation.agreement import agreement, agreement_table
from box2boxcam.evaluation.detect import coreml_confidence_floor, is_coreml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_detector


def dets(*frames):
    """``dets([(box, conf), ...], ...)`` -> detection list of numpy arrays."""
    out = []
    for frame in frames:
        boxes = np.array([b for b, _ in frame], float).reshape(-1, 4)
        out.append((boxes, np.array([c for _, c in frame], float)))
    return out


BOX = [0, 0, 10, 10]


class AgreementTest(unittest.TestCase):
    def test_identical_models_agree_fully(self):
        a = dets([(BOX, 0.9)], [], [([20, 20, 30, 30], 0.4)])
        row = agreement(a, a, conf=0.25)
        self.assertEqual((row["matched"], row["only_a"], row["only_b"]), (2, 0, 0))
        self.assertEqual(row["identical_frames"], 3)
        self.assertEqual(row["mean_iou"], 1.0)
        self.assertEqual(row["max_abs_conf_diff"], 0.0)

    def test_shifted_box_and_confidence_difference(self):
        a = dets([(BOX, 0.90)])
        b = dets([([1, 0, 11, 10], 0.85)])
        row = agreement(a, b, conf=0.25)
        self.assertEqual(row["matched"], 1)
        self.assertAlmostEqual(row["mean_iou"], 90 / 110)
        self.assertAlmostEqual(row["mean_abs_conf_diff"], 0.05)

    def test_unmatched_boxes_count_per_side(self):
        a = dets([(BOX, 0.9), ([50, 50, 60, 60], 0.6)])
        b = dets([(BOX, 0.9), ([80, 80, 90, 90], 0.7)])
        row = agreement(a, b, conf=0.25)
        self.assertEqual((row["matched"], row["only_a"], row["only_b"]), (1, 1, 1))
        self.assertEqual(row["identical_frames"], 0)

    def test_threshold_applies_to_both_models(self):
        # B keeps the box just above the threshold, A just below: a disagreement.
        a = dets([(BOX, 0.24)])
        b = dets([(BOX, 0.26)])
        row = agreement(a, b, conf=0.25)
        self.assertEqual((row["matched"], row["only_a"], row["only_b"]), (0, 0, 1))
        self.assertIsNone(row["mean_iou"])
        both_kept = agreement(a, b, conf=0.2)
        self.assertEqual(both_kept["matched"], 1)

    def test_frame_count_must_match(self):
        with self.assertRaises(ValueError):
            agreement(dets([]), dets([], []), conf=0.25)

    def test_table(self):
        rows = [agreement(dets([(BOX, 0.9)]), dets([]), conf=0.25)]
        table = agreement_table(rows, "A", "B")
        self.assertIn("| 0.25 | 0 | 1 | 0 | 0 / 1 | - | - | - |", table)


class ModelLabelTest(unittest.TestCase):
    def test_stems_when_unique(self):
        labels = evaluate_detector.model_labels([Path("a/new.pt"), Path("ref.pt")])
        self.assertEqual(labels, ["new", "ref"])

    def test_names_when_an_export_shares_the_stem(self):
        paths = [Path("m/ball.mlpackage"), Path("w/ball.pt")]
        labels = evaluate_detector.model_labels(paths)
        self.assertEqual(labels, ["ball.mlpackage", "ball.pt"])

    def test_same_file_name_is_refused(self):
        with self.assertRaises(SystemExit):
            evaluate_detector.model_labels([Path("a/ball.pt"), Path("b/ball.pt")])

    def test_agreement_needs_compare(self):
        with self.assertRaises(SystemExit):
            evaluate_detector.main(
                [
                    "--weights",
                    "a.pt",
                    "--frames-dir",
                    "f",
                    "--labels",
                    "l",
                    "--agreement",
                ]
            )


class CoreMLTest(unittest.TestCase):
    def test_is_coreml(self):
        self.assertTrue(is_coreml(Path("ball.mlpackage")))
        self.assertFalse(is_coreml(Path("ball.pt")))

    def test_coreml_needs_macos(self):
        with (
            mock.patch.object(
                evaluate_detector.platform, "system", return_value="Linux"
            ),
            self.assertRaises(SystemExit),
        ):
            evaluate_detector.check_coreml(Path("ball.mlpackage"), 0.05)
        # A .pt model is never checked.
        evaluate_detector.check_coreml(Path("ball.pt"), 0.05)

    def test_warns_when_model_threshold_is_above_conf(self):
        with (
            mock.patch.object(
                evaluate_detector.platform, "system", return_value="Darwin"
            ),
            mock.patch.object(
                evaluate_detector, "coreml_confidence_floor", return_value=0.1
            ),
            mock.patch.object(evaluate_detector.sys, "stderr") as err,
        ):
            evaluate_detector.check_coreml(Path("ball.mlpackage"), 0.05)
            self.assertIn("0.1", "".join(c.args[0] for c in err.write.call_args_list))
            err.reset_mock()
            evaluate_detector.check_coreml(Path("ball.mlpackage"), 0.1)
            self.assertFalse(err.write.called)

    def test_floor_is_none_without_coremltools(self):
        with (
            mock.patch.dict(sys.modules, {"coremltools": None}),
            tempfile.TemporaryDirectory() as tmp,
        ):
            self.assertIsNone(coreml_confidence_floor(Path(tmp) / "m.mlpackage"))


if __name__ == "__main__":
    unittest.main()
