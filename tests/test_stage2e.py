"""Small synthetic CPU tests; no external images, weights or GPU benchmarks."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from modern_pca import preprocessing_cache as c
from modern_pca import numerical_parity as p
from modern_pca import reimplementation as r


@pytest.fixture
def cached(tmp_path,monkeypatch):
    extracted=tmp_path/'images'; rows=[]
    for label,code,suffix in [('benign',0,'norm'),('tumour',1,'tu')]:
        path=extracted/f'val_dataset_1_{suffix}'/'a.jpg'; path.parent.mkdir(parents=True)
        pixels=np.random.default_rng(code).integers(0,256,(609+code,612,3),dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append(dict(sample_id=f'VAL1:{label}/a.jpg',relative_path=f'{label}/a.jpg',cohort='VAL1',class_name=label,
                         ground_truth_code=code,file_sha256=r.sha256_file(path),width=609+code,height=612))
    class Identity:
        def transform(self,x): return x.copy()
    monkeypatch.setattr(c,'frozen_training_rows',lambda *args:rows)
    monkeypatch.setattr(r,'normalizers',lambda *args:(Identity(),Identity()))
    before={str(x):x.read_bytes() for x in extracted.rglob('*.jpg')}
    output=tmp_path/'cache'; meta=c.build_subset_cache(rows,extracted,output,shard_size=1)
    yield rows,extracted,output,meta
    assert all(Path(path).read_bytes()==value for path,value in before.items())


def test_exact_cache_and_online_parity(cached):
    rows,extracted,output,meta=cached; reader=c.CacheReader(output)
    for i,row in enumerate(rows):
        path=r.image_path(row,extracted)
        online=r.preprocess_legacy(row,path,r.normalizers(None))
        assert c.exact_comparison(c.to_float(reader.get(i)),online)==dict(pass_=True,max_absolute_difference=0.,differing_elements=0)
    assert all(item['dtype']=='uint8' and item['split']=='train' and item['shape']==[350,350,3] for item in meta['items'])
    assert meta['provenance']['split_sha256']==c.SPLIT_HASHES
    assert meta['provenance']['stain_reference_sha256']==r.REFERENCE_SHA


def test_cache_resume_and_context_binding(cached):
    rows,extracted,output,meta=cached
    hashes={x.name:r.sha256_file(x) for x in output.glob('*.npy')}
    c.build_subset_cache(rows,extracted,output,shard_size=1)
    assert hashes=={x.name:r.sha256_file(x) for x in output.glob('*.npy')}
    with pytest.raises(ValueError,match='context mismatch'):
        c.build_subset_cache(list(reversed(rows)),extracted,output,shard_size=1)


def test_corrupt_tensor_and_mapping_rejected(cached):
    rows,extracted,output,meta=cached
    path=output/'cache.json'; altered=json.loads(path.read_text()); altered['items'][0]['source_jpeg_sha256']='0'*64
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='mapping'):
        c.CacheReader(output)
    path.write_text(json.dumps(meta))
    data=np.load(output/meta['items'][0]['shard'],mmap_mode='r+'); data[0,0,0,0]^=1; data.flush(); del data
    with pytest.raises(ValueError,match='checksum'):
        c.CacheReader(output)


def test_full_cache_and_val2_forbidden(cached):
    rows,extracted,output,meta=cached
    with pytest.raises(ValueError,match='full-cohort'):
        c.build_subset_cache([rows[0]]*1025,extracted,output/'full')
    bad=dict(rows[0],sample_id='VAL2:benign/a.jpg',cohort='VAL2')
    with pytest.raises(ValueError,match='VAL1-training'):
        c.build_subset_cache([bad],extracted,output/'val2')


@pytest.mark.parametrize('parallel,prefetch',[(1,0),(1,2),(2,2),(4,2)])
def test_loading_order_values_and_augmentation(cached,parallel,prefetch):
    rows,extracted,output,meta=cached; reader=c.CacheReader(output)
    order=c.epoch_order(reader.items,3)
    assert order==c.epoch_order(reader.items,3)
    actual=[]
    for images,labels,ids in c.dataset(reader,order,batch_size=1,parallel=parallel,prefetch=prefetch,epoch=3):
        for image,label,i in zip(images.numpy(),labels.numpy(),ids.numpy()):
            expected=np.asarray(r.augment(c.to_float(reader.get(i)),rows[i]['sample_id'],3))
            np.testing.assert_array_equal(image,expected)
            assert label==rows[i]['ground_truth_code']; actual.append(int(i))
    assert actual==order and sorted(actual)==[0,1]


def test_storage_exact_header_bytes(cached):
    _,_,output,meta=cached
    assert c.storage_requirement((1,1),1)['npy_bytes']==sum(x.stat().st_size for x in output.glob('*.npy'))
    full=c.storage_requirement()
    assert full['images']==145819 and full['shards']==570
    assert full['raw_tensor_bytes']==145819*350*350*3


def test_difference_robust_chunking_and_fixed_tolerances():
    x=np.arange(23,dtype=np.float32)/10; y=x+np.float32(1e-7)
    a=p.difference(x,y,chunk=3); b=p.difference(x,y,chunk=100)
    for key in a: assert a[key]==pytest.approx(b[key])
    assert p.gradient_pass(a) and p.weights_pass(a,1)
    assert not p.gradient_pass(p.difference(np.zeros(3,np.float32),np.ones(3,np.float32)))
    with pytest.raises(ValueError,match='Nonfinite'): p.difference(np.array([np.nan]),np.array([0.]))
    with pytest.raises(ValueError,match='shape'): p.difference(np.zeros(3),np.zeros(2))


def test_cpu_gpu_forward_utility():
    a=dict(probabilities=[[.8,.2],[.1,.9]],logits=[[2.,.1],[.2,2.]],loss=.12)
    assert p.forward_comparison(a,a)['pass']
    b=dict(a,probabilities=[[.2,.8],[.1,.9]])
    assert not p.forward_comparison(a,b)['pass']
    near=dict(a,probabilities=[[.8-1e-7,.2+1e-7],[.1,.9]],loss=.1200001)
    assert p.forward_comparison(a,near)['pass']


def test_deterministic_representative_subset():
    rows=[dict(sample_id=f'VAL1:{label}/{i}',cohort='VAL1',class_name=label,width=609 if i%20==0 else 610,height=612)
          for label,count in [('benign',700),('tumour',400)] for i in range(count)]
    a=c.representative_rows(rows); b=c.representative_rows(list(reversed(rows)))
    assert a==b and len(a)==1000 and len({x['sample_id'] for x in a})==1000
    assert {x['width'] for x in a}=={609,610} and {x['class_name'] for x in a}=={'benign','tumour'}


def test_forecast_accounting_and_explicit_planning_margin():
    from tools.report_stage2e import estimate
    stages=[dict(boundary='fixture',epochs=14,train_images_per_second=10.,validation_images_per_second=20.)]
    result=estimate(stages,.2,12.,100.)
    assert result['training_seconds']==pytest.approx(14*116655/10)
    assert result['validation_seconds']==pytest.approx(14*29164/20)
    assert result['cache_generation_seconds']==pytest.approx(145819*.2)
    assert result['expected_complete_seconds']==pytest.approx(145819*.2+14*(116655/10+29164/20)+112)
    assert result['conservative_complete_seconds'][1]>=result['expected_complete_seconds']


def test_component_diagnostic_keeps_weights_unchanged():
    import tensorflow as tf
    from modern_pca.profile_reimplementation import diagnostic_components, compiled_components
    model=tf.keras.Sequential([tf.keras.Input((2,)),tf.keras.layers.Dense(2,activation='softmax')])
    accumulator=r.Accumulator(model,r.adam(1e-5))
    before=[v.numpy().copy() for v in model.weights]
    result=diagnostic_components(model,accumulator,np.ones((4,2),np.float32),np.array([0,1,0,1]))
    assert set(result['seconds'])=={'forward','loss','backward','gradient_accumulation'}
    assert all(np.isfinite(x) and x>=0 for x in result['seconds'].values())
    assert all(np.array_equal(a,b.numpy()) for a,b in zip(before,model.weights))
    compiled=compiled_components(model,accumulator,np.ones((4,2),np.float32),np.array([0,1,0,1]))
    assert compiled['status']=='PASS'
    assert all(np.isfinite(x) and x>=0 for x in compiled['seconds'].values())
    assert all(np.array_equal(a,b.numpy()) for a,b in zip(before,model.weights))
