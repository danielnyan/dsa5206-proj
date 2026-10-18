"""DSA5206 follow-up Experiment 2: patient-disjoint SICAPv2 conservative late-backbone transfer.

Run from repo root with python -B -u -m modern_pca.vanda_sicap_partial.
This is NOT a reproduction of the paper's training protocol.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import reimplementation as r
from .vanda_distributed_training import distribute, make_train_fn, make_checkpoint, digest, write_json_atomic
from .vanda_sicapv2_distributed import gpu_setup, preprocess_batch

ROOT = Path('/scratch/e1536052/DSA5206')
PARENT = ROOT / 'vanda_runs/nasnet_production_14epoch/epoch14.keras'
IMAGES = ROOT / 'sicap_transfer/images'
TRAIN = Path('reports/stage2j_followup/sicap_train_patients.csv')
DEV = Path('reports/stage2j_followup/sicap_dev_patients.csv')
SUMMARY = Path('reports/stage2j_followup/split_summary.json')
PARENT_SHA = '433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde'
TRAIN_SHA = 'd599edd2a541e962925ba69658ee3678e812e88a9622452e37638421a4e5785f'
DEV_SHA = '660273c4ce1959b77c4b35e39bbe713309d14f921fe62f8012c2f3334e43245a'
TOTAL = 8126
BATCH = 100
LR = 1e-7
BOUNDARY = "normal_conv_1_17"


def read_inputs():
    for path, expected in ((PARENT,PARENT_SHA),(TRAIN,TRAIN_SHA),(DEV,DEV_SHA)):
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f'SHA256 mismatch: {path}: {actual}')
    summary = json.loads(SUMMARY.read_text())
    if summary.get('train_manifest_sha256') != TRAIN_SHA or summary.get('development_manifest_sha256') != DEV_SHA or summary.get('patient_overlap') != 0:
        raise RuntimeError('Split summary disagrees with frozen manifests')
    with TRAIN.open(newline='') as f:
        train_rows = list(csv.DictReader(f))
    with DEV.open(newline='') as f:
        dev_rows = list(csv.DictReader(f))
    if len(train_rows) != TOTAL or len(dev_rows) != 1833:
        raise RuntimeError('Split sample counts differ from frozen protocol')
    if {x['patient_id'] for x in train_rows} & {x['patient_id'] for x in dev_rows}:
        raise RuntimeError('Patient overlap')
    if len({x['sample_id'] for x in train_rows + dev_rows}) != 9959:
        raise RuntimeError('Missing/duplicate sample IDs')
    for group, b, t in ((train_rows,3188,4938),(dev_rows,585,1248)):
        if sum(x['ground_truth_code']=='0' for x in group) != b or sum(x['ground_truth_code']=='1' for x in group) != t:
            raise RuntimeError('Class count mismatch')
        for x in group:
            if (x['partition'] != 'train' or
                x['official_label'] not in ('NC','G3','G4','G5') or
                int(x['ground_truth_code']) != int(x['official_label'] != 'NC') or
                Path(x['filename']).name != x['filename']):
                raise RuntimeError('Invalid training row')
            x['ground_truth_code'] = int(x['ground_truth_code'])
            path = IMAGES / x['filename']
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(IMAGES.resolve()):
                raise RuntimeError(f'Missing/unsafe image: {path}')
    return train_rows, dev_rows


def load_partial(strategy):
    with strategy.scope():
        model = tf.keras.models.load_model(PARENT, compile=False)
        r.validate_model(model)
        backbones = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                     and any(x.name == 'normal_conv_1_1' for x in layer.layers)]
        if len(backbones) != 1:
            raise RuntimeError('Cannot find unique NASNet backbone')
        model.backbone = backbones[0]
        before = r.weight_hashes(model.weights)
        count = r.configure_stage(model, BOUNDARY)
        if r.weight_hashes(model.weights) != before:
            raise RuntimeError('Stage configuration changed weights')
        if any(layer.trainable for layer in model.backbone.layers
               if isinstance(layer, tf.keras.layers.BatchNormalization)):
            raise RuntimeError('BatchNormalization was unfrozen')
        if not model.get_layer('paper_dense_256').trainable or not model.get_layer('class_probabilities').trainable:
            raise RuntimeError('Classifier head unexpectedly frozen')
        backbone_variables = len(model.backbone.trainable_variables)
        if backbone_variables < 1:
            raise RuntimeError('No backbone variables selected for partial training')
        if not (124896002 < count < 208800242):
            raise RuntimeError(f'Unexpected partial trainable parameter count: {count}')
        print(f'PARTIAL trainable_params={count} trainable_vars={len(model.trainable_variables)} '
              f'backbone_trainable_vars={backbone_variables} boundary={BOUNDARY} lr={LR}', flush=True)
    return model, count


def run(mode, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be absent or empty: {output}')
    rows, _ = read_inputs()
    strategy = gpu_setup()
    model, nparams = load_partial(strategy)
    with strategy.scope():
        optimizer = r.adam(LR)
        optimizer.build(model.trainable_variables)
    if int(optimizer.iterations.numpy()) != 0:
        raise RuntimeError('Optimizer not fresh')
    frozen = r.weight_hashes(model.non_trainable_variables)
    step = make_train_fn(strategy,model,optimizer)
    ref = r.normalizers(r.REFERENCE)
    started = time.perf_counter()
    consumed = 0
    weighted_loss = 0.0
    limit = BATCH if mode == 'smoke' else len(rows)
    for offset in range(0,limit,BATCH):
        group = rows[offset:offset+BATCH]
        x,y=preprocess_batch(group,IMAGES,ref)
        n=len(group)
        loss=float(step(distribute(strategy,x,n), distribute(strategy,y,n), tf.constant(n,tf.int32)).numpy())
        if not np.isfinite(loss):
            raise RuntimeError('Non-finite gradient step')
        consumed+=n
        weighted_loss+=n*loss
        updates=int(optimizer.iterations.numpy())
        if updates%10==0 or consumed==limit:
            print(f'PARTIAL {mode} samples={consumed}/{limit} updates={updates} loss={weighted_loss/consumed:.6f} elapsed_min={(time.perf_counter()-started)/60:.2f}',flush=True)
    expected_updates=math.ceil(limit/BATCH)
    if consumed!=limit or int(optimizer.iterations.numpy())!=expected_updates:
        raise RuntimeError('Unexpected update count')
    if frozen!=r.weight_hashes(model.non_trainable_variables):
        raise RuntimeError('Frozen weights mutated')
    if mode=='smoke':
        print('=== PARTIAL SMOKE PASS (1 UPDATE; NO MODEL SAVED) ===',flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    contract={'purpose':'DSA5206_experiment2_SICAPv2_late_backbone_lr1e-7',
              'starting_checkpoint_sha256':PARENT_SHA,'train_manifest_sha256':TRAIN_SHA,
              'dev_manifest_sha256':DEV_SHA,'train_samples':TOTAL,'dev_samples':1833,
              'train_patients':58,'dev_patients':16,'epoch':15,'lr':LR,'effective_batch':BATCH,
              'updates':expected_updates,'trainable_params':nparams,'backbone_frozen':False,'fine_tune_boundary':BOUNDARY,'batch_norm_frozen':True,
              'training_augmentation_epoch':15, 'VAL2_images_accessed':False,
              'trainer_sha256':digest(Path(__file__))}
    write_json_atomic(output/'contract.json',contract)
    checkpoint=make_checkpoint(model,optimizer)
    checkpoint_dir=output/'epoch15_resume'
    checkpoint_dir.mkdir()
    checkpoint.write(str(checkpoint_dir/'state'))
    model_file=output/'epoch15_partial.keras'
    model.save(model_file)
    write_json_atomic(output/'history.json',{'samples':consumed,'updates':expected_updates,
                     'loss':weighted_loss/consumed,'elapsed_seconds':time.perf_counter()-started})
    write_json_atomic(output/'metadata.json',{'completed_epochs':15,'model_sha256':digest(model_file),
                      'parent_model_sha256':PARENT_SHA,'trainable_params':nparams,
                      'purpose':'experiment2_late_backbone_lr1e-7'})
    artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    write_json_atomic(output/'run_complete.json',{'status':'PASS','artifacts':artifacts})
    print('=== PARTIAL EXPERIMENT 2 TRAINING PASS ===',flush=True)
    print('MODEL SHA256',digest(model_file),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=['smoke','train'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'val2' in str(args.output).lower():
        raise ValueError('Output path cannot reference VAL2')
    run(args.mode,args.output)


if __name__=='__main__':
    main()
