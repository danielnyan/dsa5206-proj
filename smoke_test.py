"""CPU-safe smoke checks for model construction, forward, and checkpoints."""

import argparse
import gc
import tempfile
from pathlib import Path

import torch

from MedMamba import MODEL_CONFIGS, VSSM, create_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-variants", action="store_true")
    args = parser.parse_args()

    if not args.skip_variants:
        for variant in MODEL_CONFIGS:
            model = create_model(variant, num_classes=6)
            parameters = sum(value.numel() for value in model.parameters())
            print(f"MODEL_LOAD_OK variant={variant} parameters={parameters:,}")
            del model
            gc.collect()

    # A reduced architecture exercises the identical blocks and portable scan
    # without making a CPU-only CI machine process a full 224px model.
    config = {
        "depths": [1, 1, 1, 1],
        "dims": [16, 32, 64, 128],
        "d_state": 4,
        "num_classes": 3,
    }
    model = VSSM(**config).eval()
    sample = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        output = model(sample)
    assert output.shape == (1, 3), output.shape
    assert torch.isfinite(output).all()
    print(f"CPU_FORWARD_OK input={tuple(sample.shape)} output={tuple(output.shape)}")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "wrapped.pt"
        wrapped = {"model": {f"module.{key}": value for key, value in model.state_dict().items()}}
        torch.save(wrapped, checkpoint)
        restored = VSSM(**config)
        incompatible = load_checkpoint(restored, checkpoint)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        for expected, actual in zip(model.parameters(), restored.parameters()):
            assert torch.equal(expected, actual)
        print("CHECKPOINT_LOAD_OK wrapper=model prefix=module.")

    print("MEDMAMBA_SMOKE_OK")


if __name__ == "__main__":
    main()
