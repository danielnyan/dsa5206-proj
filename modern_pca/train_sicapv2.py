"""One predefined SECONDARY external fine-tuning epoch; never a VAL1 reproduction.

This entry point has no evaluation, tuning or schedule restart path. It accepts
only an authenticated Stage 2J plan and the official SICAP training manifest.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from . import reimplementation as r, sicapv2 as s

PURPOSE = 'secondary_SICAPv2_epoch15'
MODEL_SHA = '5cecd53138d74cffc7313521a1c78544096d7e872440b496631a5666534e1e61'


def protocol():
    return dict(purpose=PURPOSE, starting_epoch=14, final_epoch=15, extra_epochs=1,
        boundary='normal_conv_1_1', learning_rate=1e-6, physical_batch=4,
        effective_batch=100, seed=42, precision='float32', TF32=False,
        BN='frozen/inference', optimizer='fresh Adam', beta_1=0.9,beta_2=0.999,
        epsilon=1e-7, weight_decay=None, clipping=None, amsgrad=False,
        loss='unweighted sparse categorical crossentropy',
        horizontal_flip=0.5,vertical_flip=0.5,epoch_for_flips=15,
        sampling='each official training sample exactly once; no resampling',
        order='SHA256 SICAPv2-epoch15-v1|42|<sample_id>; ties sample_id',
        preprocessing='frozen PIL RGB/LANCZOS350/brightness/Macenko/float32 divided by255',
        reference_sha256=r.REFERENCE_SHA, class_order=['benign','tumour'],
        validation='none; no checkpoint selection', allocator='cuda_malloc_async',
        scale='whole 512px 10x patch resized to350; explicit magnification deviation')


def source_hashes():
    from .stage2i_contract import source_hashes as primary_sources
    names = ('modern_pca/sicapv2.py','modern_pca/train_sicapv2.py',
             'modern_pca/evaluate_sicapv2.py','tools/inspect_sicapv2.py')
    return dict(primary_sources(), **{name:r.sha256_file(Path(name)) for name in names})


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('protocol') != protocol() or plan.get('source_sha256') != source_hashes():
        raise ValueError('Frozen Stage 2J protocol/source mismatch')
    if plan.get('status') not in ('GO','CONDITIONAL GO') or plan.get('preprocessing_failures') != 0:
        raise ValueError('Stage 2J assessment does not permit this experiment')
    if plan.get('software') != r.capture_environment()['software']:
        raise ValueError('Frozen Stage 2J software changed')
    return plan


def forbid_val2():
    def audit(event,args):
        if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if 'val_dataset_2' in os.fsdecode(args[0]).lower():
                raise ValueError('VAL2 images forbidden during SICAP training')
    sys.addaudithook(audit)


def training_rows(manifest, plan):
    r.sha256_file(manifest,plan['manifest_sha256'])
    with Path(manifest).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows)!=plan['samples'] or len({v['sample_id'] for v in rows})!=len(rows):
        raise ValueError('Training sample count/identity mismatch')
    for row in rows:
        label = int(row['ground_truth_code'])
        if (row['partition']!='train' or row['official_label'] not in ('NC','G3','G4','G5')
                or label!=int(row['official_label']!='NC')
                or Path(row['filename']).name!=row['filename'] or '\\' in row['filename']
                or row['sample_id']!='SICAPv2:'+row['filename']):
            raise ValueError('Invalid official SICAP training row')
        row['ground_truth_code']=label
    if rows!=s.sample_order(rows):
        raise ValueError('Frozen training order changed')
    return rows


def image_path(root,row):
    root=Path(root)
    path=root/row['filename']
    if path.is_symlink() or root.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe SICAP image path')
    return path


def load_start(path):
    import tensorflow as tf
    r.sha256_file(path,MODEL_SHA)
    model=tf.keras.models.load_model(path,compile=False)
    r.validate_model(model)
    # Deserialization retains the functional nested NASNet, not Python attributes.
    backbones=[layer for layer in model.layers if isinstance(layer,tf.keras.Model)
               and any(item.name=='normal_conv_1_1' for item in layer.layers)]
    if len(backbones)!=1: raise ValueError('Expected exactly one NASNet backbone')
    model.backbone=backbones[0]
    before=r.weight_hashes(model.weights)
    r.configure_stage(model,'normal_conv_1_1')
    if before!=r.weight_hashes(model.weights): raise ValueError('Stage setup changed trained weights')
    if any(layer.trainable for layer in model.backbone.layers if isinstance(layer,tf.keras.layers.BatchNormalization)):
        raise ValueError('BatchNormalization must remain frozen')
    return model


def fresh_optimizer(model):
    optimizer=r.adam(1e-6)
    optimizer.build(model.trainable_variables)
    if int(optimizer.iterations.numpy())!=0:
        raise ValueError('Optimizer must start fresh')
    return optimizer


def batches(rows,root,reference):
    import tensorflow as tf
    for offset in range(0,len(rows),4):
        batch=rows[offset:offset+4]
        images=[r.augment(r.preprocess_legacy(row,image_path(root,row),reference),row['sample_id'],15)
                for row in batch]
        yield tf.stack(images),tf.constant([row['ground_truth_code'] for row in batch],tf.int32),tf.range(offset,offset+len(batch))


def save_state(model,optimizer,output,plan_sha,samples):
    import tensorflow as tf
    output=Path(output); output.mkdir(exist_ok=False)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    checkpoint.save_counter
    checkpoint.write(str(output/'state'))
    state=dict(purpose=PURPOSE,completed_epochs=15,boundary='normal_conv_1_1',samples=samples,
        plan_sha256=plan_sha,optimizer_iterations=int(optimizer.iterations.numpy()),
        model_tensor_sha256=r.weight_hashes(model.weights),optimizer_tensor_sha256=r.weight_hashes(optimizer.variables),
        files_sha256={p.name:r.sha256_file(p) for p in output.glob('state.*')})
    r.write_json(output/'checkpoint.json',state)
    return state


def restore_state(model,optimizer,output,plan_sha,samples):
    import tensorflow as tf
    output=Path(output); state=json.loads((output/'checkpoint.json').read_text())
    if (state.get('purpose')!=PURPOSE or state.get('completed_epochs')!=15
            or state.get('boundary')!='normal_conv_1_1' or state.get('samples')!=samples
            or state.get('plan_sha256')!=plan_sha or state.get('optimizer_iterations')!=math.ceil(samples/100)
            or set(state['files_sha256'])!={'state.index','state.data-00000-of-00001'}):
        raise ValueError('Invalid completed secondary checkpoint')
    for name,digest in state['files_sha256'].items():
        r.sha256_file(output/name,digest)
    optimizer.build(model.trainable_variables)
    checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer); checkpoint.save_counter
    checkpoint.read(str(output/'state')).assert_consumed()
    if (r.weight_hashes(model.weights)!=state['model_tensor_sha256']
            or r.weight_hashes(optimizer.variables)!=state['optimizer_tensor_sha256']
            or int(optimizer.iterations.numpy())!=state['optimizer_iterations']):
        raise ValueError('Restored weights/Adam mismatch')
    return state


def run(args):
    forbid_val2()
    if not args.train_one_epoch or not args.accept_magnification_deviation:
        raise ValueError('Explicit one-epoch training and magnification acknowledgement required')
    if os.environ.get('TF_GPU_ALLOCATOR')!='cuda_malloc_async':
        raise ValueError('Frozen secondary protocol requires cuda_malloc_async')
    plan=read_plan(args.plan)
    if str(args.images.resolve())!=plan['images_root']:
        raise ValueError('Require the audited SICAP image root')
    rows=training_rows(args.manifest,plan)
    r.ensure_output_separation(args.images,[args.output])
    logger=r.logging_setup(args.output)
    device=r.gpu_setup()
    model=load_start(args.model)
    optimizer=fresh_optimizer(model)
    frozen=r.weight_hashes(model.non_trainable_variables)
    accumulator=r.Accumulator(model,optimizer,effective=100)
    reference=r.normalizers(r.REFERENCE)
    from .cached_training import train_batches
    result=train_batches(accumulator,batches(rows,args.images,reference))
    if (result['samples']!=len(rows) or result['updates']!=math.ceil(len(rows)/100)
            or int(optimizer.iterations.numpy())!=result['updates'] or accumulator.count
            or frozen!=r.weight_hashes(model.non_trainable_variables)):
        raise ValueError('Epoch coverage/optimizer/frozen-state mismatch')
    plan_sha=r.sha256_file(args.plan)
    save_state(model,optimizer,args.output/'epoch_15_resume',plan_sha,len(rows))
    # Strictly verify the saved trained tensors and Adam state before export.
    restore_state(model,optimizer,args.output/'epoch_15_resume',plan_sha,len(rows))
    model.save(args.output/'epoch15.keras')
    r.write_json(args.output/'history.json',dict(parent_model_sha256=MODEL_SHA,epochs=[dict(epoch=15,**result)]))
    r.write_json(args.output/'metadata.json',dict(purpose=PURPOSE,completed_epochs=15,
        plan_sha256=plan_sha,model_sha256=r.sha256_file(args.output/'epoch15.keras'),
        protocol=protocol(),device=device,training_samples=len(rows),VAL2_images_accessed=False))
    r.write_json(args.output/'run_complete.json',dict(status='PASS',purpose=PURPOSE,
        artifacts={p.name:r.sha256_file(p) for p in args.output.iterdir() if p.is_file() and p.name!='run.log'}))
    logger.info('Secondary epoch15 complete: %s',result)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','model','manifest','images','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--train-one-epoch',action='store_true')
    parser.add_argument('--accept-magnification-deviation',action='store_true')
    return run(parser.parse_args(argv))


if __name__=='__main__': raise SystemExit(main())
