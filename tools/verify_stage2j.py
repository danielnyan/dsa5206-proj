"""Stage 2J synthetic tests and zero-update GPU feasibility probe; never train."""
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import sys
import time
import zipfile

from modern_pca import reimplementation as r, train_sicapv2 as t
from tools.inspect_sicapv2 import ROOT


def guard():
    blocked=[]
    def audit(event,args):
        if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0]).replace('\\','/').lower()
            if 'val_dataset_2' in path and not path.startswith('/tmp/stage2j_tests_'):
                blocked.append(path); raise RuntimeError('Real VAL2 access forbidden')
    sys.addaudithook(audit)
    return blocked


def tests():
    import pytest
    result={}
    for name,paths in [('focused',['tests/test_sicapv2.py','tests/test_sicapv2_training.py']),('full',[])]:
        log=Path(f'reports/stage2j_{name}_tests.log')
        with log.open('w') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',f'/tmp/stage2j_tests_{name}_{time.time_ns()}',*paths])
        text=log.read_text()
        result[name]=dict(exit_code=int(code),summary=text.splitlines()[-1],log=str(log))
        print(name,result[name],flush=True)
        if code: raise RuntimeError(f'{name} tests failed')
    return result


def gpu():
    import tensorflow as tf
    import numpy as np
    if os.environ.get('TF_GPU_ALLOCATOR')!='cuda_malloc_async': raise ValueError('Require async allocator')
    def forbidden(*args,**kwargs): raise AssertionError('Optimizer updates/training iterator forbidden in assessment')
    tf.keras.optimizers.Adam.apply_gradients=forbidden
    r.Accumulator.add=forbidden; r.Accumulator.flush=forbidden; t.batches=forbidden
    device=r.gpu_setup()
    path=Path('runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras')
    model=t.load_start(path)
    expected=json.loads(Path('runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch_14_resume/checkpoint.json').read_text())['model_tensor_sha256']
    before=r.weight_hashes(model.weights)
    assert before==expected
    optimizer=t.fresh_optimizer(model)
    accumulator=r.Accumulator(model,optimizer,effective=100)
    initial_optimizer=r.weight_hashes(optimizer.variables)
    from modern_pca import sicapv2 as s
    with zipfile.ZipFile(ROOT/'SICAPv2.zip') as archive:
        official=s.table(archive.read('SICAPv2/partition/Test/Train.xlsx'),['image_name','NC','G3','G4','G5','G4C'])
    rows=s.sample_order([dict(sample_id='SICAPv2:'+v['image_name'],filename=v['image_name'],
        ground_truth_code=s.primary_label(v)[1]) for v in official])[:4]
    for row in rows: row['file_sha256']=r.sha256_file(ROOT/'extracted/SICAPv2/images'/row['filename'])
    reference=r.normalizers(r.REFERENCE)
    images=tf.stack([r.augment(r.preprocess_legacy(row,ROOT/'extracted/SICAPv2/images'/row['filename'],reference),row['sample_id'],15) for row in rows])
    labels=tf.constant([int(row['ground_truth_code']) for row in rows],tf.int32)
    tf.config.experimental.reset_memory_stats('GPU:0')
    started=time.perf_counter()
    # Read-only forward/backward feasibility. No buffer or optimizer writes.
    with tf.GradientTape() as tape:
        probabilities=model(images,training=False)
        loss=tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels,probabilities))
    gradients=tape.gradient(loss,model.trainable_variables)
    for gradient in gradients:
        assert gradient is not None
        tf.debugging.assert_all_finite(gradient,'Nonfinite feasibility gradient')
    seconds=time.perf_counter()-started
    memory=tf.config.experimental.get_memory_info('GPU:0')
    assert r.weight_hashes(model.weights)==before
    assert r.weight_hashes(optimizer.variables)==initial_optimizer and optimizer.iterations.numpy()==0
    assert accumulator.count==0 and all(np.count_nonzero(v.numpy())==0 for v in accumulator.buffers)
    r.sha256_file(path,t.MODEL_SHA)
    return dict(status='PASS',device=device,allocator='cuda_malloc_async',physical_batch=4,effective_batch=100,
        samples=[row['sample_id'] for row in rows],boundary='normal_conv_1_1',memory_bytes=memory,
        seconds=seconds,probe='one read-only forward/backward pass with fresh Adam slots and accumulation buffers allocated',
        optimizer_updates=0,training_batches=0,validation_batches=0,weights_BN_Adam_buffers_unchanged=True,
        restored_epoch14_weights_exact=True,training_started=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['tests','gpu'])
    phase=parser.parse_args().phase
    blocked=guard()
    value=globals()[phase]()
    r.write_json(Path(f'reports/stage2j_{phase}_verification.json'),dict(status='PASS',result=value,
        VAL2_images_accessed=False,blocked_VAL2_attempts=blocked,training_started=False))


if __name__=='__main__': main()
