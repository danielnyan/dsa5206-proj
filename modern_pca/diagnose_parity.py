"""Stage 2F bounded diagnosis. Reuses Stage 2E data/configuration unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from . import reimplementation as r
from . import numerical_parity as p
from . import preprocessing_cache as c
from . import profile_reimplementation as e


def read(path):
    return json.loads(Path(path).read_text())


def initialize(root,baseline):
    r.logging_setup(root)
    checks=read('reports/stage2e_checks.json')
    protected=dict(checks['protected_files_sha256'])
    protected.update({k:v for k,v in checks['source_code_and_report_sha256'].items() if k.endswith('.py')})
    for path,digest in protected.items():
        r.sha256_file(Path(path),digest)
    protocol=dict(created_at_utc=r.utc_now(),baseline=str(baseline),protected_sha256=protected,
        tolerances=p.TOLERANCES,near_zero=1e-8,relative_denominator_floor=1e-8,
        elementwise_diagnostics='abs >1e-4; relative >1e-3 using max(abs(CPU),1e-8); either means union. These diagnostics do not replace tensor-level Stage 2E criteria.',
        updates=10,physical_batch=4,effective_batch=100,probe_samples=64,
        multi_update_practicality='Stage 2E CPU updates took 26-36 seconds; ten updates per repeat are practical.',
        augmentation=False,production_training=False,VAL2_accessed=False,
        scope='Only existing 1000-image VAL1-training cache; no new preprocessing or source-image reads',
        implementation_sha256=r.sha256_file(Path(__file__)),git=r.capture_git())
    r.write_json(root/'protocol.json',protocol)
    repro=root/'reproduction'; repro.mkdir()
    for name in ('initial.weights.h5','initial.json','protocol.json','samples.json'):
        shutil.copyfile(baseline/name,repro/name)
        r.sha256_file(repro/name,r.sha256_file(baseline/name))
    shutil.copytree(baseline/'cache',repro/'cache')
    for path in (baseline/'cache').iterdir():
        if path.is_file(): r.sha256_file(repro/'cache'/path.name,r.sha256_file(path))
    items=read(baseline/'cache/cache.json')['items']
    remainder=sorted(range(100,len(items)),key=lambda i:hashlib.sha256(('stage2f-v1|'+items[i]['sample_id']).encode()).hexdigest())
    sequence=list(range(100))+remainder
    if len(sequence)!=1000 or len(set(sequence))!=1000:
        raise ValueError('Expected the exact Stage 2E 1000-sample cache')
    r.write_json(root/'sequence.json',dict(indices=sequence,sample_ids=[items[i]['sample_id'] for i in sequence]))


def check(root):
    for name,digest in read(root/'protocol.json')['protected_sha256'].items():
        r.sha256_file(Path(name),digest)


def gradient_stats(a,b,chunk=1000000):
    """Float64 diagnostics, bounded chunks; original acceptance utility unchanged."""
    basic=p.difference(a,b,chunk=chunk)
    dot=nb=relative_sum=0.; abs_count=rel_count=either_count=near_a=near_b=0
    aa=a.reshape(-1); bb=b.reshape(-1)
    for offset in range(0,len(aa),chunk):
        x=np.asarray(aa[offset:offset+chunk],np.float64); y=np.asarray(bb[offset:offset+chunk],np.float64)
        delta=np.abs(x-y); rel=delta/np.maximum(np.abs(x),1e-8)
        absolute=delta>p.TOLERANCES['gradient_max_atol']; relative=rel>p.TOLERANCES['gradient_max_rtol']
        dot+=float(np.dot(x,y)); nb+=float(np.dot(y,y)); relative_sum+=float(rel.sum())
        abs_count+=int(absolute.sum()); rel_count+=int(relative.sum()); either_count+=int((absolute|relative).sum())
        near_a+=int((np.abs(x)<=1e-8).sum()); near_b+=int((np.abs(y)<=1e-8).sum())
    n=basic['elements']; norm_b=float(np.sqrt(nb)); norm_a=basic['reference_l2']
    cosine=dot/(norm_a*norm_b) if norm_a and norm_b else (1. if norm_a==norm_b else 0.)
    basic.update(cpu_norm=norm_a,gpu_norm=norm_b,cosine_similarity=float(np.clip(cosine,-1,1)),
        mean_relative=relative_sum/max(n,1),absolute_exceed_count=abs_count,relative_exceed_count=rel_count,
        either_exceed_count=either_count,absolute_exceed_fraction=abs_count/max(n,1),
        relative_exceed_fraction=rel_count/max(n,1),either_exceed_fraction=either_count/max(n,1),
        near_zero_cpu_count=near_a,near_zero_gpu_count=near_b,near_zero_cpu_fraction=near_a/max(n,1),near_zero_gpu_fraction=near_b/max(n,1),
        strict_stage2e_pass=bool(p.gradient_pass(basic)))
    return basic


def tensor_specs(variables):
    specs=[]; offset=0
    for i,v in enumerate(variables):
        n=int(np.prod(v.shape)); specs.append(dict(index=i,name=getattr(v,'path',v.name),shape=list(v.shape),elements=n,offset=offset)); offset+=n
    return specs


def predict_probe(model,x,y):
    import tensorflow as tf
    fn=tf.function(lambda z:model(z,training=False),input_signature=[tf.TensorSpec((None,350,350,3),tf.float32)])
    def predict(indices):
        probabilities=np.concatenate([fn(x[indices[i:i+4]]).numpy() for i in range(0,len(indices),4)])
        loss=float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y[indices],probabilities)).numpy())
        return dict(probabilities=probabilities.tolist(),loss=loss,sha256=hashlib.sha256(probabilities.tobytes()).hexdigest())
    return predict


def trajectory(args):
    import tensorflow as tf
    root=args.output; target=root/args.job; r.logging_setup(target); info=e.configure(args.device)
    baseline=Path(read(root/'protocol.json')['baseline']); reader=c.CacheReader(baseline/'cache')
    model,_=e.load_model(baseline,r.SCHEDULE[0][0]); acc=r.Accumulator(model,r.adam(1e-5))
    x=np.stack([c.to_float(reader.get(i)) for i in range(1000)])
    y=np.array([z['ground_truth_code'] for z in reader.items],np.int64)
    sequence=read(root/'sequence.json')['indices']; probe=predict_probe(model,x,y)
    frozen=r.weight_hashes(model.non_trainable_variables)
    report=dict(device=info,status='RUNNING',updates=[],before=probe(list(range(64))),
                gradient_specs=tensor_specs(acc.variables),weight_specs=tensor_specs(model.weights),
                input_sha256=hashlib.sha256(memoryview(x)).hexdigest(),labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                initial_checkpoint_sha256=r.sha256_file(baseline/'initial.weights.h5'))
    if args.job=='cpu_A': e.save_vectors(root/'initial_vector.npy',model.weights)
    for update in range(1,11):
        losses=[]; counts=[]; started=time.perf_counter()
        for offset in range(0,100,4):
            indices=sequence[(update-1)*100+offset:(update-1)*100+offset+4]
            losses.append(acc.add(x[indices],y[indices])); counts.append(acc.count)
            if update==1 and offset==0 and args.job.endswith('_A'):
                e.save_vectors(target/'micro_sum.npy',acc.buffers)
        if counts!=list(range(4,101,4)): raise ValueError('Incorrect accumulated sample counts')
        gradient_hashes=r.weight_hashes(acc.buffers)
        if update==1 and args.job.endswith('_A'):
            e.save_vectors(target/'logical_sum.npy',acc.buffers)
            e.save_vectors(target/'logical_average.npy',acc.buffers,100)
        acc.flush()
        if acc.count!=0 or int(acc.optimizer.iterations.numpy())!=update: raise ValueError('Update/count mismatch')
        hashes=r.weight_hashes(model.weights)
        if args.job.endswith('_A'): e.save_vectors(target/f'weights_{update}.npy',model.weights)
        result=dict(update=update,loss=float(np.mean(losses)),sample_count=100,microbatch_counts=counts,
                    gradient_sum_sha256=gradient_hashes,weight_sha256=hashes,optimizer_iterations=update,
                    after=probe(list(range(64))),seconds=time.perf_counter()-started)
        report['updates'].append(result); r.write_json(target/'result.json',report)
        r.LOG.info('%s update %d/10 loss=%.9g elapsed=%.1fs',args.job,update,result['loss'],result['seconds'])
    report['frozen_weights_exact']=frozen==r.weight_hashes(model.non_trainable_variables)
    if not report['frozen_weights_exact']: raise ValueError('Frozen weights changed')
    if args.job.endswith('_A'): report['final_1000']=probe(list(range(1000)))
    report['status']='COMPLETE'; r.write_json(target/'result.json',report); check(root)


def accumulation_reference(args):
    """Recompute each microgradient with the unchanged accumulator, resetting it.

    Accumulate the exported increments independently in NumPy float32/float64.
    This separates gradient computation from the resource-buffer summation.
    """
    import tensorflow as tf
    root=args.output; target=root/args.job; r.logging_setup(target); e.configure(args.device)
    baseline=Path(read(root/'protocol.json')['baseline']); reader=c.CacheReader(baseline/'cache')
    model,_=e.load_model(baseline,r.SCHEDULE[0][0]); acc=r.Accumulator(model,r.adam(1e-5))
    specs=tensor_specs(acc.variables); elements=sum(x['elements'] for x in specs)
    sums32=np.lib.format.open_memmap(target/'independent_sum32.npy',mode='w+',dtype='float32',shape=(elements,))
    sums64=np.lib.format.open_memmap(target/'independent_sum64.npy',mode='w+',dtype='float64',shape=(elements,))
    sums32[:]=0; sums64[:]=0
    @tf.function
    def reset():
        for buffer in acc.buffers: buffer.assign(tf.zeros_like(buffer))
        return tf.reduce_sum(acc.buffers[-1])
    counts=[]; losses=[]
    for offset in range(0,100,4):
        x=np.stack([c.to_float(reader.get(i)) for i in range(offset,offset+4)])
        y=np.array([reader.items[i]['ground_truth_code'] for i in range(offset,offset+4)],np.int64)
        losses.append(acc.add(x,y)); counts.append(acc.count)
        for spec,buffer in zip(specs,acc.buffers):
            g=buffer.numpy().reshape(-1); start=spec['offset']; stop=start+spec['elements']
            for part in range(0,len(g),1000000):
                end=min(part+1000000,len(g))
                sums32[start+part:start+end]+=g[part:end]
                sums64[start+part:start+end]+=g[part:end].astype(np.float64)
        reset().numpy(); acc.count=0
        if offset%20==0: r.LOG.info('%s independent microgradients %d/100',args.device,offset+4)
    sums32.flush(); sums64.flush()
    if counts!=[4]*25: raise ValueError('Physical-gradient reference was not reset')
    actual=np.load(root/(args.device+'_A')/'logical_sum.npy',mmap_mode='r')
    checks=[]
    for spec in specs:
        sl=slice(spec['offset'],spec['offset']+spec['elements'])
        checks.append(dict(spec=spec,exact_float32_sum=bool(np.array_equal(actual[sl],sums32[sl])),
                           float32_sum_difference=p.difference(actual[sl],sums32[sl]),
                           float64_sum_difference=p.difference(sums64[sl],actual[sl])))
    r.write_json(target/'result.json',dict(status='COMPLETE',checks=checks,counts=counts,
        samples=100,independent_microbatches=25,loss=float(np.mean(losses)),
        scope='No optimizer update in reference pass; production remains float32; float64 used only to diagnose exported gradient reduction'))
    check(root)


def launch(root,job):
    result=(root/'reproduction'/job/'result.json') if job.startswith('parity_') else root/job/'result.json'
    if result.exists() and read(result).get('status')=='COMPLETE':
        r.LOG.info('Preserving %s',job); return
    if result.parent.exists(): raise ValueError(f'Incomplete prior attempt at {result.parent}; preserve/review before retry')
    command=[sys.executable,'-B','-m','modern_pca.diagnose_parity','--output',str(root),'--worker',job]
    with (root/(job+'.console.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: r.LOG.info('Still running %s',job)
    if process.returncode: raise RuntimeError(f'{job} failed; see its console log')


def reproduction_summary(root):
    baseline=Path(read(root/'protocol.json')['baseline']); repro=root/'reproduction'
    new=e.compare_parity(repro); old=read(baseline/'parity_comparison.json')
    failures=lambda value:[x['spec']['index'] for x in value['gradients'] if not x['pass_']]
    exact={}
    for device in ('cpu','gpu'):
        exact[device]={name:r.sha256_file(repro/('parity_'+device)/name)==r.sha256_file(baseline/('parity_'+device)/name)
                       for name in ['gradients.npy','weights_1.npy','weights_2.npy','weights_3.npy']}
    result=dict(original_failures=failures(old),reproduced_failures=failures(new),same_failures=failures(old)==failures(new),
                exact_tensor_files=exact,strict_pass=new['pass'])
    r.write_json(root/'reproduction_summary.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/stage2f_parity'))
    parser.add_argument('--baseline',type=Path,default=Path('runs/stage2e_profile'))
    parser.add_argument('--worker'); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv); root=args.output
    if args.worker:
        args.job=args.worker; args.device='cpu' if 'cpu' in args.job else 'gpu'
        if args.job.startswith('parity_'):
            return e.parity_worker(SimpleNamespace(output=root/'reproduction',job=args.job,device=args.device))
        if args.job.startswith('reference_'): return accumulation_reference(args)
        return trajectory(args)
    if args.resume:
        check(root); r.logging_setup(root/('resume_'+str(time.time_ns())))
    else: initialize(root,args.baseline)
    for job in ('parity_cpu','parity_gpu'): launch(root,job)
    result=reproduction_summary(root)
    r.LOG.info('Reproduction: %s',result)
    for job in ('cpu_A','gpu_A','cpu_B','gpu_B','reference_cpu','reference_gpu'): launch(root,job)
    check(root); r.write_json(root/'run_complete.json',dict(status='COMPLETE',timestamp=r.utc_now()))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
