"""CPU-only contract tests. No ImageNet download, GPU, or VAL2 image reads."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from modern_pca import reimplementation as r
from modern_pca import train_reimplementation as train
from modern_pca import evaluate_reimplementation as evaluate


def fixture_rows():
    return [dict(row_index=i,sample_id=f"VAL1:{label}/x{i}.jpg",cohort="VAL1",class_name=label,
                 ground_truth_code=code,relative_path=f"{label}/x{i}.jpg",file_sha256="a"*64,
                 file_size_bytes=10,width=609 if i==0 else 610,height=612,mode="RGB")
            for label,code in r.CLASSES.items() for i in range(220)]


def test_split_identity_order_and_no_overlap():
    rows = fixture_rows()
    a,b = r.split_val1(rows,{"benign":20,"tumour":20})
    assert (a,b) == r.split_val1(list(reversed(rows)),{"benign":20,"tumour":20})
    assert len(a)==400 and len(b)==40
    assert not {x['sample_id'] for x in a} & {x['sample_id'] for x in b}
    assert {x['sample_id'] for x in a+b} == {x['sample_id'] for x in rows}
    for label in r.CLASSES:
        ranked = sorted([x['sample_id'] for x in rows if x['class_name']==label],key=lambda s:(hashlib.sha256(('VAL1-internal-v1|42|'+s).encode()).hexdigest(),s))
        assert {x['sample_id'] for x in b if x['class_name']==label} == set(ranked[:20])
    assert [x['row_index'] for x in a] == list(range(400))


def test_real_val1_manifest_exact_counts_without_image_access():
    path = Path("manifests/VAL1_manifest.csv")
    if not path.exists():
        pytest.skip("Verified manifest not distributed with source")
    rows = r.read_manifest(path)
    a,b = r.split_val1(rows)
    r.validate_rows(a,"VAL1",r.TRAIN_COUNTS)
    r.validate_rows(b,"VAL1",r.VALIDATION_COUNTS)
    assert len(a)==116655 and len(b)==29164
    assert not {x['sample_id'] for x in a} & {x['sample_id'] for x in b}
    selected = train.smoke_rows(a)
    assert len(selected)==400
    assert {x['sample_id'] for x in selected} <= {x['sample_id'] for x in a}
    assert {x['class_name'] for x in selected} == set(r.CLASSES)
    assert all((x['width'],x['height'])!=(610,612) for x in selected[:2])


@pytest.mark.parametrize('change',[{'cohort':'VAL2'},{'ground_truth_code':1},{'class_name':'gland'}])
def test_split_rejects_mapping_or_val2(change):
    rows=fixture_rows()
    rows[0].update(change)
    with pytest.raises(ValueError):
        r.split_val1(rows,{'benign':20,'tumour':20})


def test_val2_guard_precedes_path_access(monkeypatch):
    row=fixture_rows()[0]
    row['cohort']='VAL2'
    monkeypatch.setattr(Path,'resolve',lambda *a,**k: pytest.fail('Touched filesystem'))
    with pytest.raises(ValueError,match='VAL2'):
        r.image_path(row,Path('missing'))
    with pytest.raises(ValueError,match='locked'):
        evaluate.run(SimpleNamespace(cohort='VAL2',final_evaluation=False))


def test_path_traversal_and_mapping(tmp_path):
    row=fixture_rows()[0]
    assert r.image_path(row,tmp_path)==tmp_path/'val_dataset_1_norm'/'x0.jpg'
    row['relative_path']='benign/../val_dataset_2_norm/x.jpg'
    with pytest.raises(ValueError):
        r.image_path(row,tmp_path)


def test_preprocessing_order(tmp_path):
    path=tmp_path/'image.jpg'
    Image.new('RGB',(609,612),(20,40,60)).save(path)
    events=[]
    class Brightness:
        def transform(self,x):
            assert x.shape==(350,350,3) and x.dtype==np.uint8
            events.append('brightness')
            return x+1
    class Macenko:
        def transform(self,x):
            assert events==['brightness']
            events.append('macenko')
            return x+1
    result=r.preprocess_legacy(dict(sample_id='fixture',file_sha256=r.sha256_file(path)),path,(Brightness(),Macenko()))
    assert events==['brightness','macenko'] and result.dtype==np.float32
    assert result.shape==(350,350,3) and np.all(result>=0) and np.all(result<=1)


def test_reference_hash_guard(tmp_path):
    path=tmp_path/'reference.jpg'
    path.write_bytes(b'wrong')
    with pytest.raises(Exception,match='SHA-256 mismatch'):
        r.normalizers(path)


def tiny_model():
    import tensorflow as tf
    inputs=tf.keras.Input((2,))
    outputs=tf.keras.layers.Dense(2,activation='softmax',kernel_initializer=tf.keras.initializers.Constant([[.2,-.1],[.3,.4]]),bias_initializer='zeros')(inputs)
    return tf.keras.Model(inputs,outputs)


@pytest.mark.parametrize('physical',[1,2,4])
def test_sample_weighted_accumulation_and_partial_flush(physical):
    import tensorflow as tf
    model,reference=tiny_model(),tiny_model()
    optimizer,reference_optimizer=r.adam(.001),r.adam(.001)
    accumulator=r.Accumulator(model,optimizer,effective=5)
    x=np.arange(16,dtype=np.float32).reshape(8,2)/10
    y=np.arange(8,dtype=np.int64)%2
    for logical in (0,5):
        for offset in range(logical,min(logical+5,8),physical):
            end=min(offset+physical,logical+5,8)
            accumulator.add(x[offset:end],y[offset:end])
        accumulator.flush()
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y[logical:logical+5],reference(x[logical:logical+5],training=False)))
        reference_optimizer.apply_gradients(zip(tape.gradient(loss,reference.trainable_variables),reference.trainable_variables))
    for actual,expected in zip(model.weights,reference.weights):
        np.testing.assert_allclose(actual.numpy(),expected.numpy(),rtol=1e-5,atol=1e-7)
    assert accumulator.count==0 and int(optimizer.iterations.numpy())==2


def test_nonfinite_gradients_abort():
    model=tiny_model()
    accumulator=r.Accumulator(model,r.adam(.001))
    with pytest.raises(Exception,match='Nonfinite'):
        accumulator.add(np.array([[np.nan,0]],np.float32),np.array([0]))


def test_augmentation_stateless_and_axis_only():
    import tensorflow as tf
    image=np.arange(36,dtype=np.float32).reshape(3,4,3)
    results=[]
    for epoch in range(20):
        x=r.augment(tf.constant(image),'VAL1:benign/x.jpg',epoch).numpy()
        np.testing.assert_array_equal(x,r.augment(tf.constant(image),'VAL1:benign/x.jpg',epoch).numpy())
        assert any(np.array_equal(x,variant) for variant in [image,image[::-1],image[:,::-1],image[::-1,::-1]])
        results.append(x.tobytes())
    assert len(set(results))==4


def stage_fixture():
    import tensorflow as tf
    inputs=tf.keras.Input((2,))
    x=tf.keras.layers.Dense(2,name='stem')(inputs)
    for index in (1,3,5,7,9,11,14,17):
        x=tf.keras.layers.Dense(2,name=f'normal_conv_1_{index}')(x)
        x=tf.keras.layers.BatchNormalization(name=f'bn_{index}')(x)
    backbone=tf.keras.Model(inputs,x)
    output=tf.keras.layers.Dense(2,activation='softmax')(backbone(inputs,training=False))
    model=tf.keras.Model(inputs,output)
    model.backbone=backbone
    return model


def test_boundaries_and_bn_freezing():
    import tensorflow as tf
    model=stage_fixture()
    first=r.configure_stage(model,'normal_conv_1_17')
    assert not model.backbone.get_layer('normal_conv_1_14').trainable
    assert model.backbone.get_layer('normal_conv_1_17').trainable
    assert all(not layer.trainable for layer in model.backbone.layers if isinstance(layer,tf.keras.layers.BatchNormalization))
    frozen=r.weight_hashes(model.non_trainable_variables)
    accumulation=r.Accumulator(model,r.adam(.001))
    accumulation.add(np.ones((2,2),np.float32),np.array([0,1]))
    accumulation.flush()
    assert frozen==r.weight_hashes(model.non_trainable_variables)
    assert r.configure_stage(model,'normal_conv_1_1')>first
    with pytest.raises(ValueError):
        r.configure_stage(model,'missing')
    model.backbone=tiny_model()
    with pytest.raises(ValueError,match='Missing'):
        r.configure_stage(model,'normal_conv_1_17')


def test_head_construction_without_large_allocation(monkeypatch):
    import tensorflow as tf
    from modern_pca import models
    def backbone(**kwargs):
        assert kwargs==dict(include_top=False,weights='imagenet',input_shape=(350,350,3))
        x=tf.keras.Input((350,350,3))
        return tf.keras.Model(x,tf.keras.layers.AveragePooling2D(350)(x))
    monkeypatch.setitem(models.BACKBONES,'nasnetlarge',backbone)
    model=models.build_classifier(2)
    assert model.output_shape==(None,2)
    assert isinstance(model.layers[-3],tf.keras.layers.Flatten)
    assert model.layers[-2].units==256 and model.layers[-2].activation.__name__=='relu'
    assert model.layers[-1].activation.__name__=='softmax'
    with pytest.raises(ValueError,match='parameter count'):
        r.validate_model(model)


def test_threshold_metrics_full_precision():
    values=np.array([[.5,.5],[.49999999,.50000001],[.8,.2],[.1,.9]])
    scores=evaluate.probabilities_2(values,4)[:,1]
    np.testing.assert_array_equal(evaluate.binary_decision(scores),[0,1,0,1])
    metrics=evaluate.calculate_binary_metrics([0,0,1,1],scores)
    assert metrics['confusion_matrix']==[[1,1],[1,1]]
    assert all(metrics[k]==.5 for k in ['accuracy','precision','recall','specificity','f1'])
    assert metrics['roc_auc']==.5
    for bad in [np.ones((4,3)),np.array([[np.nan,0]]),np.array([[.2,.3]])]:
        with pytest.raises(ValueError):
            evaluate.probabilities_2(bad,len(bad))


def test_checkpoint_roundtrip_and_portable_model(tmp_path):
    import tensorflow as tf
    model=tiny_model()
    optimizer=r.adam(.001)
    accumulation=r.Accumulator(model,optimizer)
    x=np.ones((2,2),np.float32)
    accumulation.add(x,np.array([0,1]))
    accumulation.flush()
    assert train.checkpoint_roundtrip(model,optimizer,tmp_path,x)['status']=='PASS'
    path=tmp_path/'model.keras'
    model.save(path)
    restored=tf.keras.models.load_model(path,compile=False)
    np.testing.assert_array_equal(model(x).numpy(),restored(x).numpy())


def test_smoke_checkpoint_forbidden_on_val2(tmp_path):
    path=tmp_path/'model.keras'
    path.write_bytes(b'fixture')
    sidecar=dict(class_order=['benign','tumour'],stain_reference_sha256=r.REFERENCE_SHA,
                 model_sha256=r.sha256_file(path),purpose='smoke_disposable',completed_epochs=0)
    with pytest.raises(ValueError,match='epoch-14'):
        evaluate.validate_sidecar(sidecar,path,True)


def test_explicit_mode_required():
    with pytest.raises(SystemExit):
        train.parse_args([])
    assert train.parse_args(['--freeze-split']).freeze_split
    assert sum(x[1] for x in r.SCHEDULE)==14
    assert r.SCHEDULE[0]==('normal_conv_1_17',7,1e-5)


def test_frozen_split_refuses_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'read_manifest',lambda path:fixture_rows())
    monkeypatch.setattr(r,'VALIDATION_COUNTS',{'benign':20,'tumour':20})
    monkeypatch.setattr(r,'TRAIN_COUNTS',{'benign':200,'tumour':200})
    destination=tmp_path/'split'
    first=r.freeze_split(Path('unused.csv'),destination)
    assert first==r.freeze_split(Path('unused.csv'),destination)
    path=destination/'VAL1_train.csv'
    path.write_bytes(b'drift')
    with pytest.raises(ValueError,match='Frozen split differs'):
        r.freeze_split(Path('unused.csv'),destination)
    assert path.read_bytes()==b'drift'


def test_outputs_cannot_overlap_dataset(tmp_path):
    root=tmp_path/'dataset'
    with pytest.raises(ValueError,match='overlap'):
        r.ensure_output_separation(root,[root/'new-output'])
    with pytest.raises(ValueError,match='overlap'):
        r.ensure_output_separation(root,[tmp_path])
    r.ensure_output_separation(root,[tmp_path/'runs'])


def test_imagenet_cache_checksum_uses_keras_filename(tmp_path,monkeypatch):
    cache=tmp_path/'models'
    cache.mkdir()
    path=cache/'nasnet_large_no_top.h5'
    path.write_bytes(b'fixture weights')
    monkeypatch.setenv('KERAS_HOME',str(tmp_path))
    monkeypatch.setattr(r,'capture_git',lambda:{})
    monkeypatch.setattr(r,'capture_environment',lambda:{})
    monkeypatch.setattr(r,'preprocessing_settings',lambda:{})
    assert r.metadata({},'fixture',1)['imagenet_weight_files']=={path.name:r.sha256_file(path)}
