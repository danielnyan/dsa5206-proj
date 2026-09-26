import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ConfusionMatrix.main import ConfusionMatrix
from ConfusionMatrix.model import MobileNetV2
from MedMamba import VSSM, load_checkpoint, selective_scan_ref
from grad_cam.swin_model import SwinTransformer
from grad_cam.utils import GradCAM, center_crop_img, show_cam_on_image
from grad_cam.vit_model import VisionTransformer


ROOT = Path(__file__).parents[1]


class ModelSmokeTests(unittest.TestCase):
    def test_selective_scan_reference_and_backward(self):
        u = torch.randn(2, 4, 6, requires_grad=True)
        delta = torch.randn(2, 4, 6, requires_grad=True)
        a = -torch.rand(4, 3)
        b = torch.randn(2, 2, 3, 6)
        c = torch.randn(2, 2, 3, 6)
        output = selective_scan_ref(u, delta, a, b, c, delta_softplus=True)
        self.assertEqual(output.shape, u.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_reduced_medmamba_forward_backward_and_checkpoint_wrappers(self):
        config = dict(
            depths=[1, 1, 1, 1],
            dims=[16, 32, 64, 128],
            d_state=4,
            num_classes=3,
        )
        model = VSSM(**config)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 3))
        output.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(
                {"state_dict": {f"module.{key}": value for key, value in model.state_dict().items()}},
                checkpoint,
            )
            restored = VSSM(**config)
            incompatible = load_checkpoint(restored, checkpoint)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])

    def test_reference_models_forward(self):
        with torch.inference_mode():
            mobile = MobileNetV2(num_classes=5).eval()(torch.randn(1, 3, 64, 64))
            self.assertEqual(mobile.shape, (1, 5))

            vit = VisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=64,
                depth=2,
                num_heads=4,
                num_classes=5,
            ).eval()(torch.randn(1, 3, 32, 32))
            self.assertEqual(vit.shape, (1, 5))

            swin = SwinTransformer(
                patch_size=4,
                embed_dim=32,
                depths=(1, 1, 1, 1),
                num_heads=(2, 4, 8, 16),
                window_size=4,
                num_classes=5,
            ).eval()(torch.randn(1, 3, 32, 32))
            self.assertEqual(swin.shape, (1, 5))


class UtilitySmokeTests(unittest.TestCase):
    def test_grad_cam_and_image_helpers(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
        with GradCAM(model, [model[0]]) as cam:
            mask = cam(torch.randn(1, 3, 32, 32, requires_grad=True), target_category=1)
        self.assertEqual(mask.shape, (1, 32, 32))
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        cropped = center_crop_img(image, 32)
        self.assertEqual(cropped.shape, (32, 32, 3))
        overlay = show_cam_on_image(cropped.astype(np.float32) / 255, mask[0], use_rgb=True)
        self.assertEqual(overlay.shape, (32, 32, 3))

    def test_confusion_matrix_updates(self):
        confusion = ConfusionMatrix(2, ["negative", "positive"])
        confusion.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        np.testing.assert_array_equal(confusion.matrix, np.array([[1, 0], [1, 1]]))

    def test_notebook_is_clean_json(self):
        notebook = json.loads(
            (ROOT / "notebooks" / "MedMamba_Colab_Starter.ipynb").read_text()
        )
        self.assertGreater(len(notebook["cells"]), 30)
        output_count = sum(
            len(cell.get("outputs", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertEqual(output_count, 0)


if __name__ == "__main__":
    unittest.main()
