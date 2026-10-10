"""CPU synthetic fixtures only. No production model, epoch, dataset or VAL2 read."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modern_pca import train_sicapv2 as t, sicapv2 as s, reimplementation as r
from modern_pca import evaluate_sicapv2 as e


def toy():
    import tensorflow as tf
    return tf.keras.Sequential([tf.keras.Input((2,)),tf.keras.layers.Dense(2,activation='softmax')])


def test_protocol_has_one_extra_epoch_no_schedule_restart():
    p=t.protocol()
    assert (p['starting_epoch'],p['final_epoch'],p['extra_epochs'])==(14,15,1)
    assert (p['boundary'],p['learning_rate'],p['physical_batch'],p['effective_batch'])==('normal_conv_1_1',1e-6,4,100)
    assert p['BN']=='frozen/inference' and p['reference_sha256']==r.REFERENCE_SHA


def test_fresh_adam():
    optimizer=t.fresh_optimizer(toy())
    assert int(optimizer.iterations.numpy())==0
    assert float(optimizer.learning_rate.numpy())==pytest.approx(1e-6)
    assert (optimizer.beta_1,optimizer.beta_2,optimizer.epsilon,optimizer.amsgrad)==(.9,.999,1e-7,False)
    assert optimizer.weight_decay is None and optimizer.clipnorm is None and optimizer.clipvalue is None
    assert all(np.count_nonzero(v.numpy())==0 for v in optimizer.variables[2:])


def test_strict_completed_checkpoint_restores_weights_and_adam(tmp_path):
    import tensorflow as tf
    model=toy(); opt=t.fresh_optimizer(model)
    for v in model.weights: v.assign(tf.ones_like(v)*.123)
    for v in opt.variables[2:]: v.assign(tf.ones_like(v)*.456)
    opt.iterations.assign(2)
    before=r.weight_hashes(model.weights); slots=r.weight_hashes(opt.variables)
    t.save_state(model,opt,tmp_path/'checkpoint','plan',159)
    for v in model.weights: v.assign(tf.zeros_like(v))
    opt.iterations.assign(8)
    t.restore_state(model,opt,tmp_path/'checkpoint','plan',159)
    assert r.weight_hashes(model.weights)==before and r.weight_hashes(opt.variables)==slots
    model.save(tmp_path/'epoch15.keras')
    assert r.weight_hashes(tf.keras.models.load_model(tmp_path/'epoch15.keras',compile=False).weights)==before
    with pytest.raises(ValueError): t.restore_state(model,opt,tmp_path/'checkpoint','changed',159)
    meta=tmp_path/'checkpoint/checkpoint.json'
    state=json.loads(meta.read_text()); state['completed_epochs']=14; meta.write_text(json.dumps(state))
    with pytest.raises(ValueError): t.restore_state(model,opt,tmp_path/'checkpoint','plan',159)


def test_batch_preprocessing_augmentation_and_exact_coverage(monkeypatch,tmp_path):
    rows=[dict(sample_id=f'SICAPv2:{i}.jpg',filename=f'{i}.jpg',ground_truth_code=i%2) for i in range(103)]
    calls=[]
    def preprocess(row,path,reference):
        calls.append(row['sample_id']); assert path.parent==tmp_path and reference=='same reference'
        return np.ones((2,2,3),np.float32)*row['ground_truth_code']
    epochs=[]
    monkeypatch.setattr(r,'preprocess_legacy',preprocess)
    monkeypatch.setattr(r,'augment',lambda image,sample_id,epoch:(epochs.append(epoch) or image))
    output=list(t.batches(rows,tmp_path,'same reference'))
    assert [len(v[0]) for v in output]==[4]*25+[3]
    assert calls==[row['sample_id'] for row in rows] and epochs==[15]*103
    assert [int(i) for _,_,indices in output for i in indices]==list(range(103))


def test_effective100_final_partial_group():
    import tensorflow as tf
    from modern_pca.cached_training import train_batches
    model=toy(); opt=t.fresh_optimizer(model); acc=r.Accumulator(model,opt,effective=100)
    data=[(tf.ones((4,2)),tf.zeros(4,tf.int32),tf.range(4)) for _ in range(25)]
    data.append((tf.ones((3,2)),tf.zeros(3,tf.int32),tf.range(3)))
    result=train_batches(acc,data)
    assert result['samples']==103 and result['updates']==2 and int(opt.iterations.numpy())==2 and acc.count==0


def test_epoch14_load_preserves_trained_weights_and_bn(monkeypatch,tmp_path):
    import tensorflow as tf
    inputs=tf.keras.Input((2,)); x=inputs
    for boundary,_,_ in r.SCHEDULE: x=tf.keras.layers.Dense(2,name=boundary)(x)
    x=tf.keras.layers.BatchNormalization(name='frozen_bn')(x)
    backbone=tf.keras.Model(inputs,x)
    outer=tf.keras.Input((2,)); model=tf.keras.Model(outer,tf.keras.layers.Dense(2,activation='softmax')(backbone(outer,training=False)))
    for v in model.weights: v.assign(tf.ones_like(v)*.25)
    path=tmp_path/'epoch14.keras'; model.save(path)
    monkeypatch.setattr(t,'MODEL_SHA',r.sha256_file(path))
    monkeypatch.setattr(r,'validate_model',lambda model:None)
    loaded=t.load_start(path)
    assert r.weight_hashes(loaded.weights)==r.weight_hashes(model.weights)
    assert not loaded.backbone.get_layer('frozen_bn').trainable
    before=r.weight_hashes(loaded.backbone.get_layer('frozen_bn').weights)
    loaded(tf.ones((4,2)),training=False)
    assert before==r.weight_hashes(loaded.backbone.get_layer('frozen_bn').weights)
    assert t.fresh_optimizer(loaded).iterations.numpy()==0


def test_training_manifest_rejects_test_membership(tmp_path):
    rows=s.sample_order([dict(sample_id='SICAPv2:a.jpg',filename='a.jpg',partition='test',official_label='NC',ground_truth_code=0)])
    path=tmp_path/'manifest.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError,match='official SICAP'):
        t.training_rows(path,dict(manifest_sha256=r.sha256_file(path),samples=1))


def test_training_never_calls_validation_or_evaluation():
    import ast, inspect
    tree=ast.parse(inspect.getsource(t))
    calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(name.endswith(('infer_rows','validation','read_manifest')) for name in calls)
    assert 'r.preprocess_legacy' in calls


def test_comparison_fixed_threshold_and_alignment():
    now=[dict(sample_id='a',row_index=0,ground_truth_code=1,p_tumour=.51)]
    old=[dict(sample_id='a',row_index='0',ground_truth_code='1',p_tumour='.5',predicted_binary_code='0')]
    result=e.compare(now,old)[0]
    assert result['threshold_crossing']==1 and result['delta_p_tumour']==pytest.approx(.01)
    old[0]['sample_id']='b'
    with pytest.raises(ValueError): e.compare(now,old)
