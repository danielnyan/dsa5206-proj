"""Bounded numeric diagnostics; no real model weights or external images."""
import numpy as np
import pytest
from modern_pca.diagnose_parity import gradient_stats, tensor_specs
from modern_pca import numerical_parity as original


def test_rich_gradient_statistics_match_direct_reference():
    a=np.array([0.,1e-9,1.,-2.,3.],np.float32)
    b=np.array([2e-8,0.,1.0002,-2.0001,3.],np.float32)
    stats=gradient_stats(a,b,chunk=2)
    x=a.astype(np.float64); y=b.astype(np.float64); delta=np.abs(x-y)
    relative=delta/np.maximum(np.abs(x),1e-8)
    assert stats['cpu_norm']==pytest.approx(np.linalg.norm(x))
    assert stats['gpu_norm']==pytest.approx(np.linalg.norm(y))
    assert stats['mean_relative']==pytest.approx(relative.mean())
    assert stats['cosine_similarity']==pytest.approx(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    assert stats['either_exceed_count']==int(((delta>1e-4)|(relative>1e-3)).sum())
    assert stats['near_zero_cpu_count']==2
    assert stats['strict_stage2e_pass']==original.gradient_pass(original.difference(a,b))


def test_zero_norm_and_near_zero_relative_diagnostics():
    z=np.zeros(5,np.float32)
    assert gradient_stats(z,z)['cosine_similarity']==1
    s=gradient_stats(z,np.full(5,1e-7,np.float32))
    assert s['cosine_similarity']==0
    assert s['relative_exceed_fraction']==1
    assert s['absolute_exceed_fraction']==0
    assert s['near_zero_gpu_count']==0


def test_chunking_does_not_change_verdict_or_counts():
    a=np.linspace(-.2,.4,1234,dtype=np.float32); b=a+np.float32(1e-5)
    small=gradient_stats(a,b,chunk=17); large=gradient_stats(a,b,chunk=10000)
    assert small.keys()==large.keys()
    for key in small: assert small[key]==pytest.approx(large[key])


def test_nonfinite_and_shape_mismatch_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        gradient_stats(np.array([np.nan]),np.array([0.]))
    with pytest.raises(ValueError,match='shape'):
        gradient_stats(np.zeros(2),np.zeros(3))


def test_specs_bind_names_shapes_offsets():
    class Variable:
        def __init__(self,name,shape): self.name=name; self.path=name; self.shape=shape
    specs=tensor_specs([Variable('conv/kernel',(2,3)),Variable('dense/bias',(4,))])
    assert specs[0]==dict(index=0,name='conv/kernel',shape=[2,3],elements=6,offset=0)
    assert specs[1]['offset']==6 and specs[1]['elements']==4


def test_repeatability_checks_all_updates_and_identity():
    import copy
    from tools.report_stage2f import compare_controls
    run=dict(input_sha256='inputs',labels_sha256='labels',initial_checkpoint_sha256='weights',before={'loss':1.},updates=[])
    for i in range(1,11):
        run['updates'].append(dict(update=i,loss=.1*i,gradient_sum_sha256=['gradient'],weight_sha256=['weights'],
                                   after={'sha256':'probabilities'},microbatch_counts=list(range(4,101,4)),optimizer_iterations=i))
    assert compare_controls(run,copy.deepcopy(run))['exact']
    changed=copy.deepcopy(run); changed['updates'][5]['gradient_sum_sha256']=['different']
    assert not compare_controls(run,changed)['exact']
    changed=copy.deepcopy(run); changed['updates'].pop()
    assert not compare_controls(run,changed)['exact']


def test_update_normalization_uses_update_not_model_magnitude():
    from tools.report_stage2f import update_scale
    before=np.array([100.,200.],np.float32)
    cpu=before+np.array([1.,2.],np.float32); gpu=before+np.array([1.1,2.],np.float32)
    s=update_scale(cpu,gpu,before,before,chunk=1)
    assert s['cpu_update_l2']==pytest.approx(np.sqrt(5))
    assert s['update_relative_difference']==pytest.approx(np.linalg.norm(cpu-gpu)/np.sqrt(5))
