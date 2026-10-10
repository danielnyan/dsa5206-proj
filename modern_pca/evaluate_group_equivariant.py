"""Evaluate a fixed group-equivariant checkpoint on all VAL2 patches.

No training, threshold tuning, augmentation, or checkpoint selection is performed.
Use only trusted checkpoints: these contain Python training state. This is an
external evaluation of a binary reimplementation, not the original paper model.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import escnn
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from . import reimplementation as r
from .group_equivariant_model import D8MultiScaleResNetGAP


REPOSITORY = Path(__file__).resolve().parents[1]
PREPROCESSING = "RGB-bilinear350-minus1-plus1"


def load_checkpoint(path, device):
    payload = Path(path).read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    progress, contract = saved["progress"], saved["contract"]
    if progress["epoch"] < 1 or progress["next_batch"] != 0:
        raise ValueError("Use a checkpoint saved after a completed epoch")
    expected = {
        "model_sha256": hashlib.sha256(
            (REPOSITORY / "modern_pca/group_equivariant_model.py").read_bytes()
        ).hexdigest(),
        "torch": str(torch.__version__),
        "escnn": escnn.__version__,
        "preprocessing": PREPROCESSING,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Checkpoint mismatch for {key}; use the training code/environment")
    torch.manual_seed(42)
    model = D8MultiScaleResNetGAP()
    # Explicit train() deletes escnn's constructor-created filter buffers.
    # The trainer saves in this mode. Strict loading must precede eval(), which
    # reconstructs those buffers from the loaded, trained coefficients.
    model.train()
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    model.eval()
    return model, saved, hashlib.sha256(payload).hexdigest()


def read_image(row, path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["file_sha256"]:
        raise ValueError(f"Image SHA-256 mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        tensor = TF.pil_to_tensor(image.convert("RGB"))
    tensor = TF.resize(tensor, [350, 350],
                       interpolation=InterpolationMode.BILINEAR, antialias=True)
    return tensor.float() / 127.5 - 1.0


def calculate_metrics(truth, predicted, scores, loss_sum):
    truth, predicted, scores = map(np.asarray, (truth, predicted, scores))
    if (truth.ndim != 1 or not len(truth) or predicted.shape != truth.shape
            or scores.shape != truth.shape or not np.isfinite(scores).all()
            or not np.isin(truth, [0, 1]).all()
            or not np.isin(predicted, [0, 1]).all()
            or np.any((scores < 0) | (scores > 1)) or not np.isfinite(loss_sum)):
        raise ValueError("Invalid labels, predictions, probabilities, or loss")
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())

    def ratio(a, b):
        return a / b if b else None

    return {
        "loss": float(loss_sum / len(truth)),
        "accuracy": (tp + tn) / len(truth),
        "precision": ratio(tp, tp + fp),
        "recall_sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
        "confusion_matrix": matrix.tolist(),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def run(args):
    started = time.perf_counter()
    output = args.output.resolve()
    r.ensure_output_separation(args.data, [output])
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the GPU environment")
    device = torch.device(args.device)
    print("Checking checkpoint...", flush=True)
    model, saved, checkpoint_sha = load_checkpoint(args.checkpoint, device)
    manifest = REPOSITORY / "manifests/VAL2_manifest.csv"
    print("Checking authenticated VAL2 manifest and image paths...", flush=True)
    rows = r.read_manifest(manifest, cohort="VAL2")
    paths = [r.image_path(row, args.data, final_evaluation=True) for row in rows]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

    output.mkdir(parents=True, exist_ok=False)
    temporary_csv = output / "predictions.csv.partial"
    truth, predicted, scores = [], [], []
    loss_sum = 0.0
    print(f"Evaluating {len(rows)} images on {device}; batch size {args.batch_size}", flush=True)
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "ground_truth", "prediction", "p_tumour"])
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start:start + args.batch_size]
                inputs = torch.stack([
                    read_image(row, path)
                    for row, path in zip(batch_rows, paths[start:start + args.batch_size])
                ]).to(device)
                labels = torch.tensor([row["ground_truth_code"] for row in batch_rows],
                                      dtype=torch.long, device=device)
                logits = model(inputs)
                if logits.shape != (len(batch_rows), 2) or not torch.isfinite(logits).all().item():
                    raise ValueError("Invalid or nonfinite model output")
                loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                # Preserve the trainer's argmax rule, including benign on ties.
                batch_predictions = logits.argmax(dim=1).cpu().tolist()
                batch_scores = logits.softmax(dim=1)[:, 1].cpu().tolist()
                for row, pred, score in zip(batch_rows, batch_predictions, batch_scores):
                    label = row["ground_truth_code"]
                    truth.append(label)
                    predicted.append(pred)
                    scores.append(score)
                    writer.writerow([row["sample_id"], label, pred, score])
                done = len(truth)
                if done // 1000 != start // 1000 or done == len(rows):
                    stream.flush()
                    print(f"Completed {done}/{len(rows)} images", flush=True)

    metrics = calculate_metrics(truth, predicted, scores, loss_sum)
    report = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "VAL2", "evaluation": "fixed_checkpoint_external_evaluation",
        "completed_training_epochs": saved["progress"]["epoch"],
        "samples": len(truth), "class_order": ["benign", "tumour"],
        "class_counts": {"benign": truth.count(0), "tumour": truth.count(1)},
        "positive_class": "tumour",
        "confusion_matrix_axes": "rows=true, columns=predicted",
        "decision_rule": "argmax logits; ties select benign",
        "undefined_metric_value": "null", "preprocessing": PREPROCESSING,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
        "model_code_sha256": saved["contract"]["model_sha256"],
        "training_contract": saved["contract"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_image_sha256_verified": True,
        "torch": str(torch.__version__), "escnn": escnn.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "elapsed_seconds_before_final_export": time.perf_counter() - started,
        "metrics": metrics,
    }
    temporary_csv.replace(output / "predictions.csv")
    temporary_json = output / "metrics.json.partial"
    temporary_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary_json.replace(output / "metrics.json")
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    print(f"Evaluation complete. Results: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Extracted root containing val_dataset_2_norm and val_dataset_2_tu")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted training checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
