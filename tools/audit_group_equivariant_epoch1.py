"""Independent full-VAL1 validation audit; no training or VAL2 image access.

Reads the saved split CSVs directly, independently counts predictions and loss,
and checks them against the checkpoint's epoch history. Outputs are separate
from the training run. Run as a module from the repository root.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from modern_pca.group_equivariant_model import D8MultiScaleResNetGAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=Path('runs/group_equivariant_run01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Choose a new audit output directory'
    assert torch.cuda.is_available(), 'CUDA is required for this full audit'
    torch.set_num_threads(2)
    started = time.perf_counter()
    checkpoint = args.run / 'epoch1.pt'
    payload = checkpoint.read_bytes()
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    assert saved['progress']['epoch'] == 1 and saved['progress']['next_batch'] == 0
    repository = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((repository / 'modern_pca/group_equivariant_model.py').read_bytes()).hexdigest() == saved['contract']['model_sha256']
    split_rows = []
    split_hashes = {}
    for name in ('VAL1_train.csv', 'VAL1_internal_validation.csv'):
        blob = (args.run / 'split' / name).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == saved['contract']['split'][name]
        assert blob == (repository / 'manifests/VAL1_internal_v1' / name).read_bytes()
        split_rows.append(list(csv.DictReader(io.StringIO(blob.decode()))))
        split_hashes[name] = digest
    train, validation = split_rows
    assert len(train) == 116655 and len(validation) == 29164
    overlaps = {}
    for key in ('sample_id', 'relative_path', 'file_sha256'):
        a, b = ({row[key] for row in rows} for rows in split_rows)
        assert len(a) == len(train) and len(b) == len(validation)
        overlaps[key] = len(a & b)
        assert not overlaps[key], f'Cross-split overlap: {key}'
    assert Counter(row['class_name'] for row in validation) == {'benign': 18602, 'tumour': 10562}
    print('Split audit passed; no overlapping IDs, paths or manifest content hashes.', flush=True)

    def read_tensor(row):
        label = {'benign': 0, 'tumour': 1}[row['class_name']]
        assert row['cohort'] == 'VAL1' and int(row['ground_truth_code']) == label
        prefix, relative = row['relative_path'].split('/', 1)
        assert prefix == row['class_name']
        assert row['sample_id'] == 'VAL1:' + row['relative_path']
        root = args.data / ('val_dataset_1_norm' if label == 0 else 'val_dataset_1_tu')
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve())
        image_bytes = path.read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == row['file_sha256'], path
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = TF.pil_to_tensor(image.convert('RGB'))
        image = TF.resize(image, [350, 350], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return image.float() / 127.5 - 1.0

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = D8MultiScaleResNetGAP()
    model.train()
    model.load_state_dict(saved['model'], strict=True)
    model.cuda().eval()
    state_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    args.output.mkdir(parents=True)
    records = []
    confusion = np.zeros((2, 2), dtype=np.int64)
    losses = []
    with torch.inference_mode():
        for offset in range(0, len(validation), 4):
            rows = validation[offset:offset + 4]
            images = torch.stack([read_tensor(row) for row in rows]).cuda()
            logits = model(images)
            assert logits.shape == (len(rows), 2) and torch.isfinite(logits).all()
            # Independent per-image NLL and direct comparison of class logits.
            log_probabilities = logits.double().log_softmax(dim=1).cpu().numpy()
            batch_scores = logits.softmax(dim=1)[:, 1].cpu().tolist()
            batch_predictions = (logits[:, 1] > logits[:, 0]).long().cpu().tolist()
            for i, (row, pred, score) in enumerate(zip(rows, batch_predictions, batch_scores)):
                true = int(row['ground_truth_code'])
                confusion[true, pred] += 1
                losses.append(-float(log_probabilities[i, true]))
                records.append([row['sample_id'], true, pred, score])
            if len(records) % 2000 == 0 or len(records) == len(validation):
                print(f'Audited {len(records)}/{len(validation)}', flush=True)

    assert all(torch.equal(value.cpu(), state_before[key]) for key, value in model.state_dict().items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    truth = np.array([row[1] for row in records])
    predictions = np.array([row[2] for row in records])
    correct = int((truth == predictions).sum())
    assert correct == int(confusion.trace())
    accuracy = correct / len(records)
    history = saved['progress']['history'][0]
    expected_correct = round(history['validation_accuracy'] * len(validation))
    tn, fp, fn, tp = map(int, confusion.ravel())
    result = {
        'checkpoint_sha256': checkpoint_sha,
        'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split_hashes': split_hashes,
        'cross_split_manifest_overlaps': overlaps,
        'all_validation_image_sha256_verified': True,
        'samples': len(records), 'correct': correct, 'incorrect': len(records) - correct,
        'confusion_matrix': confusion.tolist(),
        'matrix_axes': 'rows=true, columns=predicted; benign,tumour',
        'accuracy': accuracy, 'loss_float64_logsoftmax': float(np.mean(losses)),
        'logged_accuracy': history['validation_accuracy'], 'logged_loss': history['validation_loss'],
        'logged_correct': expected_correct,
        'exact_correct_count_match': correct == expected_correct,
        'absolute_loss_difference': abs(float(np.mean(losses)) - history['validation_loss']),
        'tumour_precision': tp / (tp + fp) if tp + fp else None,
        'tumour_recall': tp / (tp + fn), 'specificity': tn / (tn + fp),
        'majority_class_accuracy': float(np.mean(truth == 0)),
        'shuffled_label_accuracy_seed42': float(np.mean(np.random.default_rng(42).permutation(truth) == predictions)),
        'adam_step_values': sorted({int(state['step'].item()) for state in saved['optimizer']['state'].values()}),
        'model_state_unchanged_during_validation': True,
        'checkpoint_file_unchanged': True,
        'elapsed_seconds': time.perf_counter() - started,
        'limitation': 'No patient/slide identities: this audit cannot rule out correlated patches or near duplicates across splits.',
    }
    with (args.output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'ground_truth', 'prediction', 'p_tumour'])
        writer.writerows(records)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
