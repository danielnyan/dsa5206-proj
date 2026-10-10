"""CPU tests with synthetic images/checkpoints; no actual VAL2 images are read."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import escnn
from PIL import Image
import torch

from modern_pca import evaluate_group_equivariant as e


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.checkpoint = cls.root / "fixture.pt"
        model = e.D8MultiScaleResNetGAP()
        model.train()
        cls.saved = {
            "model": model.state_dict(),
            "progress": {"epoch": 1, "next_batch": 0},
            "contract": {
                "model_sha256": hashlib.sha256(
                    (e.REPOSITORY / "modern_pca/group_equivariant_model.py").read_bytes()
                ).hexdigest(),
                "torch": str(torch.__version__), "escnn": escnn.__version__,
                "preprocessing": e.PREPROCESSING,
            },
        }
        torch.save(cls.saved, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_training_checkpoint_strict_load_and_eval_buffers(self):
        model, saved, _ = e.load_checkpoint(self.checkpoint, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(sum(k.endswith(".filter") for k in model.state_dict()), 12)
        for key, value in saved["model"].items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_known_confusion_matrix_and_auc(self):
        metrics = e.calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1], [.1, .8, .4, .9], 2.)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        for key in ("loss", "accuracy", "precision", "recall_sensitivity", "specificity", "f1"):
            self.assertEqual(metrics[key], .5)
        self.assertEqual(metrics["roc_auc"], .75)

    def test_undefined_precision_and_nonfinite_scores(self):
        metrics = e.calculate_metrics([0, 1], [0, 0], [.1, .2], 1.)
        self.assertIsNone(metrics["precision"])
        with self.assertRaises(ValueError):
            e.calculate_metrics([0, 1], [0, 1], [.1, float("nan")], 1.)

    def test_tampered_image_rejected(self):
        path = self.root / "bad.jpg"
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            e.read_image({"file_sha256": "0" * 64}, path)

    def test_synthetic_end_to_end_export_and_no_overwrite(self):
        data = self.root / "data"
        rows = []
        for code, (label, directory) in enumerate((("benign", "norm"), ("tumour", "tu"))):
            relative = f"{directory}/fixture.jpg"
            path = data / f"val_dataset_2_{directory}" / relative
            path.parent.mkdir(parents=True)
            Image.new("RGB", (610, 612), (40 + code * 80, 100, 160)).save(path)
            rows.append({"sample_id": f"VAL2:{label}/{relative}",
                         "relative_path": f"{label}/{relative}", "cohort": "VAL2",
                         "class_name": label, "ground_truth_code": code,
                         "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        args = SimpleNamespace(data=data, checkpoint=self.checkpoint,
                               output=self.root / "results", batch_size=1, device="cpu")
        before = self.checkpoint.read_bytes()
        with patch.object(e.r, "read_manifest", return_value=rows):
            report = e.run(args)
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["class_counts"], {"benign": 1, "tumour": 1})
        self.assertEqual(json.loads((args.output / "metrics.json").read_text()), report)
        self.assertEqual(len((args.output / "predictions.csv").read_text().splitlines()), 3)
        self.assertFalse(list(args.output.glob("*.partial")))
        self.assertEqual(before, self.checkpoint.read_bytes())
        with self.assertRaises(FileExistsError):
            e.run(args)


if __name__ == "__main__":
    unittest.main()
