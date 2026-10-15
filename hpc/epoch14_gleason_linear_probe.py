#!/usr/bin/env python3
"""True frozen-backbone linear probing of saved PANDA NASNet embeddings.

Inputs are produced by epoch14_gleason_fewshot.py (protocol v2); the original
probe and its feature files are NEVER modified. Train one multinomial, L2-
regularized logistic regression readout for every {F44,F22,F11} x {5,50} x seed.
Labeled support samples are exactly those recorded in support_selection_audit.csv.
No NASNet inference, TensorFlow, GPU, training-bank 1NN or validation tuning.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn

STAGES = ("F44", "F22", "F11")
CLASSES = ("GP3", "GP4", "GP5")
SOURCE_PROTOCOL = "epoch14-panda-gp-transfer-v2"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, data):
    p = Path(path)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(p)

def read_population(path, expected_sha):
    """Read population in its original ordering, matching stored .npy rows."""
    ids, labels, wsis, contents = [], [], [], []
    with Path(path).open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ids.append(r['sample_id'])
            labels.append(int(r['label']))
            wsis.append(r['source_wsi'])
            contents.append(r['file_sha256'])
    actual = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if actual != expected_sha:
        raise ValueError(f'Population/order identity mismatch: {path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate sample IDs in {path}')
    if any(y not in (0, 1, 2) for y in labels):
        raise ValueError(f'Unexpected grade label in {path}')
    return {'ids': ids, 'labels': np.asarray(labels, dtype=np.int8),
            'wsis': wsis, 'contents': contents}

def supports_from_audit(path, train, requested_seeds):
    indices = {name: i for i, name in enumerate(train['ids'])}
    supports = defaultdict(list)
    with Path(path).open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            seed = int(r['seed'])
            if seed not in requested_seeds:
                continue
            i = indices.get(r['sample_id'])
            if i is None:
                raise ValueError(f'Support sample not in population: {r["sample_id"]}')
            y = int(r['label'])
            if y != int(train['labels'][i]):
                raise ValueError('Audited support label does not match population label')
            if r['file_sha256'] != train['contents'][i] or r['source_wsi'] != train['wsis'][i]:
                raise ValueError('Audited support content/WSI mismatch')
            rank = int(r['shot_rank'])
            if r['included_in_5shot'].lower() != str(rank <= 5).lower() or r['included_in_50shot'].lower() != 'true':
                raise ValueError('Audit shot-membership flags inconsistent with rank')
            supports[(seed, y)].append((rank, i))
    result = {}
    for seed in requested_seeds:
        full = []
        for y in (0, 1, 2):
            pair = sorted(supports[(seed, y)])
            if [r for r, _ in pair] != list(range(1, 51)):
                raise ValueError(f'Missing or duplicated support ranks: seed={seed} GP{y+3}')
            full.extend(i for _, i in pair)
        if len(set(full)) != 150 or len({train['contents'][i] for i in full}) != 150:
            raise ValueError(f'Duplicate support samples/content in seed={seed}')
        # Preserve the 5-as-subset-of-50 relationship without reselecting images.
        for shots in (5, 50):
            choice = [i for y in (0, 1, 2) for _, i in sorted(supports[(seed, y)])[:shots]]
            if len(choice) != 3 * shots:
                raise ValueError('Bad support count')
            result[(seed, shots)] = np.asarray(choice, dtype=np.int64)
    return result

def l2_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError('Nonfinite or zero-norm feature vectors')
    return x / norm

def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    n = int(cm.sum())
    return {'n_evaluation': n, 'accuracy': float(tp.sum() / n),
            'balanced_accuracy': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'confusion': cm.tolist(),
            'class_precision': precision.tolist(), 'class_recall': recall.tolist(),
            'class_f1': f1.tolist(), 'class_support': support.astype(int).tolist()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Original completed epoch14-GP-probe output directory')
    p.add_argument('--output', type=Path, required=True, help='New, empty output directory')
    p.add_argument('--C', type=float, default=100., help='FIXED L2 inverse regularization, not tuned on validation')
    p.add_argument('--max-iter', type=int, default=1000)
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42, help='Initial support seed (existing audit only)')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--skip-prediction-csv', action='store_true', help='Still writes compact full prediction matrices')
    p.add_argument('--preflight', action='store_true', help='Verify identity, support records, feature shapes, then exit')
    a = p.parse_args()
    if a.C <= 0 or a.max_iter <= 0 or a.chunk_size <= 0 or a.repeats <= 0:
        p.error('C, max-iter, chunk-size and repeats must be positive')
    src = a.source.resolve()
    cfg = load_json(src/'run_config.json')
    done = load_json(src/'result.json')
    ident = load_json(src/'features'/'identity.json')
    if not done.get('complete') or cfg.get('protocol') != SOURCE_PROTOCOL or ident.get('version') != SOURCE_PROTOCOL:
        raise ValueError('Only a completed protocol-v2 feature cache is supported')
    if cfg['checkpoint_sha256'] != ident['checkpoint_sha256'] or cfg['prepared_manifest_hashes'] != ident['prepared_manifest_hashes'] or cfg['population_sample_id_sha256'] != ident['population_sample_id_sha256']:
        raise ValueError('Run config and feature cache identity disagree')
    for split in ('train', 'validation'):
        prog = load_json(src/'features'/f'{split}_progress.json')
        if int(prog['completed']) != int(prog['total']):
            raise ValueError(f'{split} features are incomplete; do not read partial memmaps')
    train = read_population(src/'population_train.csv', cfg['population_sample_id_sha256']['train'])
    val = read_population(src/'population_validation.csv', cfg['population_sample_id_sha256']['validation'])
    if set(train['wsis']) & set(val['wsis']):
        raise ValueError('Train/validation WSI overlap')
    if set(train['contents']) & set(val['contents']):
        raise ValueError('Train/validation image-byte identity overlap')
    seeds = list(range(a.seed, a.seed + a.repeats))
    if any(s not in set(done['population']['support_seeds']) for s in seeds):
        raise ValueError('Requested seeds were not recorded in original experiment')
    supports = supports_from_audit(src/'support_selection_audit.csv', train, set(seeds))
    features = {}
    for stage in STAGES:
        if stage not in ident['dimensions']:
            raise ValueError(f'Missing feature stage {stage}')
        dim = ident['dimensions'][stage]
        for split, pop in (('train', train), ('validation', val)):
            path = src/'features'/f'{split}_{stage}.npy'
            x = np.load(path, mmap_mode='r', allow_pickle=False)
            if x.shape != (len(pop['ids']), dim) or x.dtype != np.float32:
                raise ValueError(f'Feature cache shape/dtype mismatch: {path}: {x.shape} {x.dtype}')
            features[(stage, split)] = x
    print(f'PREFLIGHT PASS: {len(train["ids"])} train; {len(val["ids"])} validation; {len(seeds)} matched seeds; 3 stages', flush=True)
    if a.preflight:
        return
    if a.output.resolve() == src or a.output.resolve() in src.parents or src in a.output.resolve().parents:
        raise ValueError('Output must be distinct from original features')
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    config = {'protocol': 'epoch14-linear-probe-v1', 'source': str(src),
              'checkpoint_sha256': cfg['checkpoint_sha256'],
              'source_run_config_sha256': digest(src/'run_config.json'),
              'source_feature_identity_sha256': digest(src/'features'/'identity.json'),
              'source_support_audit_sha256': digest(src/'support_selection_audit.csv'),
              'code_sha256': digest(Path(__file__)), 'sklearn_version': sklearn.__version__,
              'np_version': np.__version__, 'C': a.C, 'tol': a.tol, 'max_iter': a.max_iter,
              'chunk_size': a.chunk_size, 'seed_list': seeds, 'shots_per_class': [5, 50],
              'feature_transform': 'unit L2 norm, no fitting of feature scaler',
              'classifier': 'multinomial logistic regression, L2, lbfgs, intercept',
              'validation_usage': 'evaluation only; no hyperparameter selection',
              'backbone': 'frozen; saved features read only', 'classes': list(CLASSES)}
    save_json(out/'run_config.json', config)
    results = []
    yval = val['labels'].astype(np.int32)
    predictions_dir = out/'predictions'
    predictions_dir.mkdir()
    for stage in STAGES:
        print(f'=== {stage}: fitting {2 * len(seeds)} linear classifiers ===', flush=True)
        xt = features[(stage, 'train')]
        xv = features[(stage, 'validation')]
        models = []
        for shots in (5, 50):
            for seed in seeds:
                ids = supports[(seed, shots)]
                x = l2_rows(xt[ids])
                y = train['labels'][ids].astype(np.int32)
                if not np.array_equal(np.bincount(y, minlength=3), np.full(3, shots)):
                    raise ValueError('Incorrect support class distribution')
                clf = LogisticRegression(C=a.C, solver='lbfgs',
                                         fit_intercept=True, max_iter=a.max_iter, tol=a.tol)
                clf.fit(x, y)
                if not np.array_equal(clf.classes_, [0,1,2]):
                    raise RuntimeError('Unexpected classifier classes')
                models.append((shots, seed, clf))
                print(f'FIT {stage} shots={shots} seed={seed} n_iter={int(np.max(clf.n_iter_))}', flush=True)
        # Single validation pass per feature depth for all 40 classifiers.
        weights = np.concatenate([clf.coef_.astype(np.float32) for _, _, clf in models], axis=0)
        intercept = np.concatenate([clf.intercept_.astype(np.float32) for _, _, clf in models])
        n_models = len(models)
        all_preds = np.lib.format.open_memmap(predictions_dir/f'{stage}_labels.npy',
                     mode='w+', dtype=np.int8, shape=(len(yval), n_models))
        confusions = np.zeros((n_models, 3, 3), dtype=np.int64)
        for start in range(0, len(yval), a.chunk_size):
            stop = min(start + a.chunk_size, len(yval))
            batch = l2_rows(xv[start:stop])
            scores = (batch @ weights.T + intercept).reshape(stop-start, n_models, 3)
            pred = scores.argmax(axis=2).astype(np.int8)
            all_preds[start:stop] = pred
            yy = yval[start:stop]
            for j in range(n_models):
                confusions[j] += np.bincount(yy * 3 + pred[:,j].astype(np.int32), minlength=9).reshape(3, 3)
            if start == 0 or stop == len(yval) or (start // a.chunk_size) % 16 == 0:
                print(f'PREDICT {stage} {stop}/{len(yval)}', flush=True)
        all_preds.flush()
        if len(yval) != sum(confusions[0].ravel()):
            raise RuntimeError('Prediction count inconsistency')
        for j, (shots, seed, clf) in enumerate(models):
            m = metrics_from_confusion(confusions[j])
            results.append({'stage': stage, 'method': f'linear_{shots}', 'shots_per_class': shots,
                            'seed': seed, 'n_iter': int(np.max(clf.n_iter_)), **m})
            print(f'RESULT {stage} {shots}-shot seed={seed} macro_F1={m["macro_f1"]:.5f} balanced_acc={m["balanced_accuracy"]:.5f}', flush=True)
        # Human-inspectable predictions for the first seed, both budgets.
        if not a.skip_prediction_csv:
            for shots in (5, 50):
                j = next(i for i,(s,k,_) in enumerate(models) if s == shots and k == seeds[0])
                path = predictions_dir/f'{stage}_linear_{shots}_seed{seeds[0]}.csv'
                with path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(('query_sample_id','true_label','true_class','predicted_label','predicted_class','query_wsi','query_file_sha256'))
                    for i, yhat in enumerate(all_preds[:,j]):
                        yt = int(yval[i]); yhat=int(yhat)
                        writer.writerow((val['ids'][i],yt,CLASSES[yt],yhat,CLASSES[yhat],val['wsis'][i],val['contents'][i]))
        del all_preds
    # Original per-seed metrics provide a strictly paired benchmark against prototypes.
    old = {}
    with (src/'metrics_by_seed.csv').open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['method'].startswith('prototype_'):
                old[(r['stage'],int(r['shots_per_class']),int(r['seed']))] = float(r['macro_f1'])
    for r in results:
        key = (r['stage'],r['shots_per_class'],r['seed'])
        if key not in old:
            raise ValueError('Missing matched prototype result ' + repr(key))
        r['prototype_macro_f1'] = old[key]
        r['delta_macro_f1_vs_prototype'] = r['macro_f1'] - old[key]
    metrics_csv = out/'metrics_by_seed.csv'
    fields = ['stage','method','shots_per_class','seed','n_evaluation','n_iter','accuracy','balanced_accuracy',
              'macro_f1','precision_GP3','precision_GP4','precision_GP5','recall_GP3','recall_GP4','recall_GP5',
              'f1_GP3','f1_GP4','f1_GP5','prototype_macro_f1','delta_macro_f1_vs_prototype']
    with metrics_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results:
            w.writerow({**{k: r[k] for k in ('stage','method','shots_per_class','seed','n_evaluation','n_iter','accuracy','balanced_accuracy','macro_f1','prototype_macro_f1','delta_macro_f1_vs_prototype')},
                        **{f'{kind}_{c}': r['class_'+kind][i] for kind in ('precision','recall','f1') for i,c in enumerate(CLASSES)}})
    summary = []
    for stage in STAGES:
        for shots in (5, 50):
            subset = [r for r in results if r['stage'] == stage and r['shots_per_class'] == shots]
            f1 = np.array([r['macro_f1'] for r in subset], dtype=float)
            delta = np.array([r['delta_macro_f1_vs_prototype'] for r in subset], dtype=float)
            ba = np.array([r['balanced_accuracy'] for r in subset], dtype=float)
            summary.append({'stage': stage, 'method': f'linear_{shots}', 'shots_per_class': shots,
                'n_support_seeds': len(subset), 'n_evaluation': len(yval),
                'macro_f1_mean': float(f1.mean()), 'macro_f1_sd': float(f1.std(ddof=1)),
                'balanced_accuracy_mean': float(ba.mean()), 'balanced_accuracy_sd': float(ba.std(ddof=1)),
                'paired_delta_vs_prototype_macro_f1_mean': float(delta.mean()),
                'paired_delta_vs_prototype_macro_f1_sd': float(delta.std(ddof=1)),
                'number_of_seeds_beating_prototype': int(np.sum(delta > 0))})
    with (out/'summary_6_conditions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    save_json(out/'metrics_detailed.json', results)
    save_json(out/'result.json', {'complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                                 'six_conditions': summary, 'config': config,
                                 'validation_is_internal_not_external_test': True,
                                 'support_seeds_are_not_independent_validation_sets': True})
    print('SUCCESS:', out/'summary_6_conditions.csv', flush=True)

if __name__ == '__main__':
    main()
