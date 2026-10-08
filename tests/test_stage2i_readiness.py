"""Stage 2I tests use synthetic data only; no released VAL2 image paths."""
import csv
import io
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from modern_pca import evaluate_reimplementation as e, reimplementation as r
from modern_pca import stage2i_contract as f


def test_threshold_and_unrounded_auc():
    scores=np.array([.5,.50000001,.49999999,.9,.2])
    assert e.binary_decision(scores).tolist()==[0,1,0,1,0]
    metrics=e.calculate_binary_metrics([0,0,1,1,0],scores)
    assert metrics['confusion_matrix']==[[2,1],[1,1]]
    assert metrics['accuracy']==.6
    assert metrics['precision']==metrics['recall']==metrics['sensitivity']==metrics['f1']==.5
    assert metrics['specificity']==2/3
    assert metrics['exact_threshold_ties']==1
    close=[.50000001,.50000002]
    assert e.calculate_binary_metrics([0,1],close)['roc_auc']==1
    assert e.calculate_binary_metrics([0,1],np.round(close,6))['roc_auc']==.5


def test_probability_export_roundtrip():
    p=float(np.nextafter(np.float32(.5),np.float32(1)))
    values=e.probabilities_2([[1-p,p]],1)
    payload=e.csv_bytes([dict(p_benign=float(values[0,0]),p_tumour=float(values[0,1]))],['p_benign','p_tumour'])
    row=next(csv.DictReader(io.StringIO(payload.decode())))
    assert float(row['p_tumour'])==p and float(row['p_benign'])==1-p
    assert e.binary_decision(float(row['p_tumour']))==1


@pytest.mark.parametrize('batch',[1,2,4])
def test_once_inference_no_augmentation_bn_weights_and_determinism(batch,monkeypatch):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    inputs=tf.keras.Input((2,))
    outputs=tf.keras.layers.Dense(2,activation='softmax')(tf.keras.layers.BatchNormalization()(inputs))
    model=tf.keras.Model(inputs,outputs)
    before=r.weight_hashes(model.weights)
    rows=[dict(sample_id=f'VAL1:benign/{i}.jpg',row_index=i,class_name='benign',ground_truth_code=0) for i in range(7)]
    seen=[]; calls=[]
    def preprocess(row,path,reference):
        assert path==Path(str(row['row_index']))
        seen.append(row['sample_id'])
        return np.array([row['row_index']/10,.2],np.float32)
    def infer(x,training):
        calls.append((len(x),training))
        assert training is False
        return model(x,training=training)
    monkeypatch.setattr(r,'preprocess_legacy',preprocess)
    monkeypatch.setattr(r,'augment',lambda *a,**k:pytest.fail('Augmentation invoked'))
    args=(infer,rows,[Path(str(i)) for i in range(7)],None,batch,'/CPU:0',e.Timings(),logging.getLogger('test'),10)
    first,scores=e.infer_rows(*args)
    assert seen==[row['sample_id'] for row in rows]
    assert [p['sample_id'] for p in first]==seen
    assert sum(size for size,_ in calls)==7 and len(calls)==(7+batch-1)//batch
    second,_=e.infer_rows(*args)
    assert first==second and r.weight_hashes(model.weights)==before
    assert all(p['p_tumour']==score for p,score in zip(first,scores))


@pytest.mark.parametrize('invalid',['duplicate','class','count'])
def test_row_contract_rejected_before_model_or_preprocess(invalid,monkeypatch):
    rows=[dict(sample_id='VAL1:benign/a.jpg',row_index=0,class_name='benign',ground_truth_code=0)]
    paths=[Path('not-opened')]
    if invalid=='duplicate': rows*=2; paths*=2
    if invalid=='class': rows[0]['ground_truth_code']=1
    if invalid=='count': paths=[]
    monkeypatch.setattr(r,'preprocess_legacy',lambda *a:pytest.fail('Read image before rejecting contract'))
    with pytest.raises(ValueError):
        e.infer_rows(None,rows,paths,None,1,'/CPU:0',e.Timings(),None,1)


@pytest.fixture
def frozen_args(tmp_path,monkeypatch):
    args=SimpleNamespace(cohort='VAL2',batch_size=1,device='gpu',readiness=tmp_path/'readiness.json')
    monkeypatch.delenv('TF_GPU_ALLOCATOR',raising=False)
    for name in ('model','metadata','manifest','stain_reference'):
        path=tmp_path/name; path.write_text(name); setattr(args,name,path)
    report=dict(status='PASS',frozen_contract=f.scientific_contract(),source_sha256=f.source_hashes(),
                software=r.capture_environment()['software'],
                input_sha256={name:r.sha256_file(getattr(args,name)) for name in ('model','metadata','manifest','stain_reference')})
    args.readiness.write_text(json.dumps(report))
    return args,report,dict(model_sha256=report['input_sha256']['model'])


def test_frozen_readiness_exact_match(frozen_args):
    args,report,sidecar=frozen_args
    assert f.validate_readiness(args,sidecar)==report


@pytest.mark.parametrize('change',['status','threshold','class','preprocessing','source','software','model','manifest','batch','device','allocator','missing'])
def test_frozen_contract_drift_rejected(frozen_args,change,monkeypatch):
    args,report,sidecar=frozen_args
    if change=='status': report['status']='FAIL'
    if change=='threshold': report['frozen_contract']['threshold']=.4
    if change=='class': report['frozen_contract']['class_order'].reverse()
    if change=='preprocessing': report['frozen_contract']['preprocessing']['scale']='1/127.5'
    if change=='source': report['source_sha256']['modern_pca/evaluate_reimplementation.py']='bad'
    if change=='software': report['software']['tensorflow']='bad'
    if change in ('model','manifest'): getattr(args,change).write_text('changed')
    if change=='batch': args.batch_size=2
    if change=='device': args.device='cpu'
    if change=='allocator': monkeypatch.setenv('TF_GPU_ALLOCATOR','cuda_malloc_async')
    args.readiness.write_text(json.dumps(report))
    if change=='missing': args.readiness=None
    with pytest.raises(ValueError): f.validate_readiness(args,sidecar)


def test_output_must_be_fresh(tmp_path):
    with pytest.raises(FileExistsError): r.logging_setup(tmp_path)


def test_final_sidecar_scientific_contract(tmp_path):
    model=tmp_path/'model.keras'; model.write_bytes(b'synthetic')
    contract=f.scientific_contract()
    sidecar=dict(class_order=['benign','tumour'],stain_reference_sha256=r.REFERENCE_SHA,
        model_sha256=r.sha256_file(model),purpose='production_epoch14',completed_epochs=14,
        model=dict(backbone='NASNetLarge',weights='imagenet',input=[350,350,3],head=contract['head'],parameters=r.PARAMETERS),
        precision='float32',preprocessing=contract['preprocessing'])
    e.validate_sidecar(sidecar,model,True)
    for field,value in [('class_order',['tumour','benign']),('completed_epochs',13),('precision','float16'),('model',{}),('preprocessing',{})]:
        broken=deepcopy(sidecar); broken[field]=value
        with pytest.raises(ValueError): e.validate_sidecar(broken,model,True)


def test_final_lock_before_image_paths(frozen_args,monkeypatch):
    args,_,sidecar=frozen_args
    args.final_evaluation=True
    args.readiness=None
    monkeypatch.setattr(r,'read_manifest',lambda *a:[])
    args.metadata.write_text(json.dumps(sidecar))
    monkeypatch.setattr(e,'validate_sidecar',lambda *a:None)
    monkeypatch.setattr(r,'image_path',lambda *a,**k:pytest.fail('VAL2 image path touched'))
    with pytest.raises(ValueError,match='readiness'): e.run(args)


def test_evaluator_outputs_and_checkpoint_immutability(tmp_path,monkeypatch):
    import tensorflow as tf
    inputs=tf.keras.Input((2,))
    model=tf.keras.Model(inputs,tf.keras.layers.Softmax()(inputs))
    model_path=tmp_path/'model.keras'; model.save(model_path)
    model_sha=r.sha256_file(model_path)
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps(dict(class_order=['benign','tumour'],stain_reference_sha256=r.REFERENCE_SHA,
                                        model_sha256=model_sha,split={},precision='float32')))
    metadata_sha=r.sha256_file(metadata)
    manifest=tmp_path/'manifest.csv'; manifest.write_text('synthetic manifest')
    rows=[dict(sample_id=f'VAL1:{label}/{i}.jpg',row_index=i,class_name=label,ground_truth_code=code)
          for i,(label,code) in enumerate([('benign',0),('tumour',1),('benign',0)])]
    seen=[]
    def preprocess(row,path,reference):
        seen.append(row['sample_id'])
        return np.array([1-row['ground_truth_code'],row['ground_truth_code']],np.float32)
    monkeypatch.setattr(r,'read_manifest',lambda *a:rows)
    monkeypatch.setattr(r,'image_path',lambda row,*a,**k:Path(str(row['row_index'])))
    monkeypatch.setattr(r,'preprocess_legacy',preprocess)
    monkeypatch.setattr(r,'normalizers',lambda *a:None)
    monkeypatch.setattr(r,'validate_model',lambda *a:None)
    monkeypatch.setattr(r,'metadata',lambda *a:dict(fixture=True))
    args=SimpleNamespace(cohort='VAL1',final_evaluation=False,model=model_path,metadata=metadata,
        manifest=manifest,extracted=tmp_path/'source',output=tmp_path/'evaluation',device='cpu',
        stain_reference=r.REFERENCE,batch_size=2,log_level='INFO',log_every=1)
    assert e.run(args)==0
    assert seen==[row['sample_id'] for row in rows]
    metrics=json.loads((args.output/'metrics.json').read_text())
    assert metrics['metrics']['confusion_matrix']==[[2,0],[0,1]]
    for name in ('predictions.csv','confusion_matrix.json','timing_summary.json','environment.json','run_complete.json'):
        assert (args.output/name).is_file()
    assert r.sha256_file(model_path)==model_sha and r.sha256_file(metadata)==metadata_sha
    with pytest.raises(FileExistsError): e.run(args)
