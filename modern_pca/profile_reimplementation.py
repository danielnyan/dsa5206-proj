"""Stage 2E bounded profiling and numerical parity; no production training mode."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import numpy as np
from . import reimplementation as r
from . import preprocessing_cache as c
from . import numerical_parity as p


class Utilization:
    """0.5-second nvidia-smi and Linux /proc CPU samples; not a stall profiler."""
    def __enter__(self):
        self.samples=[]; self.stop=threading.Event(); self.previous=None
        self.started=time.perf_counter()
        self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
        return self
    def _loop(self):
        while not self.stop.is_set():
            try:
                values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
                total=sum(values); idle=values[3]+values[4]
                process=float(sum(os.times()[:2])); now=time.perf_counter()
                cpu=None; process_cpu=None
                if self.previous:
                    a,b,t,proc=self.previous
                    cpu=100*(1-(idle-b)/max(1,total-a)); process_cpu=100*(process-proc)/max(now-t,1e-8)
                self.previous=(total,idle,now,process)
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=3).splitlines()[0]
                gpu,memory=map(float,raw.split(','))
                self.samples.append(dict(elapsed=now-self.started,gpu_percent=gpu,gpu_memory_mib=memory,cpu_system_percent=cpu,cpu_process_percent_one_core=process_cpu))
            except (OSError,ValueError,subprocess.SubprocessError):
                pass
            self.stop.wait(.5)
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(timeout=4)
    def report(self):
        if not self.samples:
            return dict(available=False)
        means={key:float(np.mean([s[key] for s in self.samples if s[key] is not None])) for key in
               ('gpu_percent','cpu_system_percent','cpu_process_percent_one_core') if any(s[key] is not None for s in self.samples)}
        return dict(available=True,samples=len(self.samples),mean=means,peak_gpu_memory_mib=max(s['gpu_memory_mib'] for s in self.samples),
                    low_gpu_utilization_fraction=float(np.mean([s['gpu_percent']<10 for s in self.samples])),
                    low_gpu_definition='sampled utilization <10%; proxy for idle, not measured GPU stall time',trace=self.samples)


def configure(device,asynchronous=False):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    if device=='cpu':
        tf.config.set_visible_devices([], 'GPU')
        tf.config.experimental.enable_op_determinism()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.keras.utils.set_random_seed(42)
        info=dict(device='CPU',tensorflow=tf.__version__,visible_gpus=[])
    else:
        info=r.gpu_setup(); info['device']='GPU:0'
    tf.config.experimental.set_synchronous_execution(not asynchronous)
    info.update(asynchronous=asynchronous,intra_op_threads=8,inter_op_threads=2)
    return info


def check_originals(root):
    expected=json.loads((root/'protocol.json').read_text())['protected_sha256']
    for filename,digest in expected.items():
        r.sha256_file(Path(filename),digest)


def initialize_run(args):
    r.logging_setup(args.output)
    protected=[Path('modern_pca/evaluate_paper.py'),Path('modern_pca/reimplementation.py'),Path('modern_pca/train_reimplementation.py'),
               Path('manifests/VAL1_manifest.csv'),*[Path('manifests/VAL1_internal_v1')/n for n in c.SPLIT_HASHES]]
    protocol=dict(created_at_utc=r.utc_now(),samples=1000,seed=42,physical_batch=4,effective_batch=100,
                  dtype='float32',TF32=False,training_logical_batches=4,parity_updates=3,
                  cpu_multi_update_limit_seconds=900,tolerances=p.TOLERANCES,tolerance_rationale=p.RATIONALE,
                  protected_sha256={str(x):r.sha256_file(x) for x in protected},
                  stages=r.SCHEDULE,scope='VAL1 training subsets only; no full cache; no production training',
                  extracted=str(args.extracted),metadata=r.metadata({},'stage2e_bounded_profile',4),
                  implementation_sha256={x.name:r.sha256_file(x) for x in [Path(__file__),Path(c.__file__),Path(p.__file__)]})
    r.write_json(args.output/'protocol.json',protocol)
    return protocol


def format_benchmarks(reader,output):
    import tensorflow as tf
    output.mkdir(); arrays=np.stack([reader.get(i) for i in range(len(reader.items))]); n=len(arrays)
    individual=output/'individual_npy'; individual.mkdir()
    start=time.perf_counter()
    for i,array in enumerate(arrays):
        np.save(individual/f'{i:04d}.npy',array,allow_pickle=False)
    write_individual=time.perf_counter()-start
    raw=output/'raw_uint8.bin'; start=time.perf_counter(); arrays.tofile(raw); write_raw=time.perf_counter()-start
    record=output/'raw_uint8.tfrecord'; start=time.perf_counter()
    with tf.io.TFRecordWriter(str(record)) as writer:
        for array in arrays:
            writer.write(array.tobytes())
    write_record=time.perf_counter()-start
    records=list(tf.data.TFRecordDataset([str(record)]).as_numpy_iterator())
    if len(records)!=n or any(not np.array_equal(np.frombuffer(x,np.uint8).reshape(c.SHAPE),arrays[i]) for i,x in enumerate(records)):
        raise ValueError('TFRecord round-trip failed')
    del records
    mapped=np.memmap(raw,mode='r',dtype=np.uint8,shape=arrays.shape)
    def record_read(index):
        with record.open('rb') as stream:
            stream.seek(index*(c.SAMPLE_BYTES+16)+12)
            return np.frombuffer(stream.read(c.SAMPLE_BYTES),np.uint8).reshape(c.SHAPE).copy()
    loaders={'individual_npy':lambda i:np.load(individual/f'{i:04d}.npy',allow_pickle=False),
             'sharded_npy':reader.get,'raw_mmap':lambda i:np.array(mapped[i],copy=True),'tfrecord_indexed':record_read}
    sizes={'individual_npy':sum(x.stat().st_size for x in individual.iterdir()),
           'sharded_npy':sum(x.stat().st_size for x in reader.root.glob('*.npy')),
           'raw_mmap':raw.stat().st_size,'tfrecord_indexed':record.stat().st_size}
    results={}
    order=c.epoch_order(reader.items)
    for name,load in loaders.items():
        for i in range(n):
            if not np.array_equal(load(i),arrays[i]):
                raise ValueError(f'{name} cache parity failed')
        result=dict(bytes=sizes[name],exact_parity=True,read_scope='warm filesystem cache; copied complete arrays; TFRecord random reads use explicit fixed-record offsets')
        for mode,indices in [('sequential',list(range(n))),('shuffled',order)]:
            durations=[]
            for repeat in range(3):
                start=time.perf_counter(); checksum=0
                for i in indices:
                    array=load(i); checksum+=int(array[0,0,0])
                durations.append(time.perf_counter()-start)
            result[mode]=dict(seconds=durations,images_per_second=n/float(np.median(durations)))
        results[name]=result
    results['write_seconds']=dict(individual_npy=write_individual,raw_mmap=write_raw,tfrecord=write_record)
    return results


def prepare(args):
    import tensorflow as tf
    r.logging_setup(args.output/'prepare')
    configure('gpu')
    rows=c.representative_rows(c.frozen_training_rows(),1000)
    (args.output/'samples.json').write_text(json.dumps(rows),encoding='utf-8')
    reference=r.normalizers(r.REFERENCE)
    pending=[]
    def observe(pixels,row,timings):
        start=time.perf_counter(); floats=pixels.astype(np.float32); timings['float32_conversion']+=time.perf_counter()-start
        start=time.perf_counter(); floats=floats/np.float32(255.); timings['scaling']+=time.perf_counter()-start
        start=time.perf_counter()
        with tf.device('/CPU:0'):
            floats=r.augment(tf.convert_to_tensor(floats),row['sample_id'],1).numpy()
        timings['augmentation']+=time.perf_counter()-start; pending.append(floats)
        if len(pending)==4:
            start=time.perf_counter(); batch=np.stack(pending); timings['batching']+=time.perf_counter()-start
            start=time.perf_counter()
            with tf.device('/GPU:0'):
                gpu=tf.convert_to_tensor(batch); tf.reduce_sum(gpu).numpy()
            timings['host_to_GPU_transfer']+=time.perf_counter()-start
            pending.clear()
    with Utilization() as monitor:
        meta=c.build_subset_cache(rows,args.extracted,args.output/'cache',component_observer=observe)
    reader=c.CacheReader(args.output/'cache')
    parity=[]
    # Includes every explicitly selected class/dimension stratum (first ten) and both classes.
    for i in range(64):
        row=rows[i]; path=r.image_path(row,args.extracted); captured=[]
        class Capture:
            def transform(self,pixels):
                result=reference[1].transform(pixels); captured.append(result.copy()); return result
        online=r.preprocess_legacy(row,path,(reference[0],Capture()))
        uint8=c.exact_comparison(captured[0],reader.get(i)); scaled=c.exact_comparison(online,c.to_float(reader.get(i)))
        if not uint8['pass_'] or not scaled['pass_']:
            r.write_json(args.output/'cache_parity_failure.json',dict(sample_id=row['sample_id'],uint8=uint8,float32=scaled))
            raise ValueError('Exact cache parity failed; stop before cache benchmarking')
        parity.append(dict(sample_id=row['sample_id'],uint8=uint8,float32=scaled))
    for row in rows:
        r.sha256_file(r.image_path(row,args.extracted),row['file_sha256'])
    r.LOG.info('Exact online/cache parity PASS; source JPEG hashes unchanged')
    result=dict(cache_generation=meta,serial_preprocessing_utilization=monitor.report(),parity=parity,
                source_immutability='1000 source JPEG hashes rechecked after generation',full_VAL1_storage=c.storage_requirement(),
                formats=format_benchmarks(reader,args.output/'formats'))
    r.write_json(args.output/'preprocessing.json',result)
    n=1000
    result['post_cache_components_seconds']={k:meta['components_seconds'][k] for k in
        ['float32_conversion','scaling','augmentation','batching','host_to_GPU_transfer']}
    result['component_scope']='Components measured in one serial online pass; H2D includes scalar-reduction synchronization. Cache writer wall also includes hashing, logging and writes. Cache bytes are captured BEFORE scaling/augmentation; observer operates on float32 copies.'
    r.write_json(args.output/'preprocessing.json',result)
    return input_benchmarks(args)


def input_benchmarks(args):
    import tensorflow as tf
    result=json.loads((args.output/'preprocessing.json').read_text())
    reader=c.CacheReader(args.output/'cache')
    rows=json.loads((args.output/'samples.json').read_text()); n=len(rows)
    # Compare deterministic loader options, including requested overlap.
    options=[]; indices=c.epoch_order(reader.items)
    expected=[None]*n
    with tf.device('/CPU:0'):
        for i in indices:
            expected[i]=hashlib.sha256(np.asarray(r.augment(c.to_float(reader.get(i)),rows[i]['sample_id'],1)).tobytes()).hexdigest()
    for parallel,prefetch,device in [(1,0,False),(1,2,False),(2,2,False),(4,2,False),(4,2,True)]:
        settings=dict(parallel=parallel,prefetch=prefetch,device_prefetch=device)
        start=time.perf_counter(); actual=[]
        for images,labels,ids in c.dataset(reader,indices,**settings):
            values=images.numpy(); labs=labels.numpy(); indexes=ids.numpy()
            for array,label,i in zip(values,labs,indexes):
                if hashlib.sha256(array.tobytes()).hexdigest()!=expected[i] or int(label)!=rows[i]['ground_truth_code']:
                    raise ValueError('Optimized input values/labels/augmentation changed')
                actual.append(int(i))
        if actual!=indices:
            raise ValueError('Optimized order/coverage changed')
        seconds=time.perf_counter()-start
        options.append(dict(settings=settings,seconds=seconds,images_per_second=n/seconds,exact_parity=True,
                            scope='Includes host materialization, hashing and exact-order checks; tune using subsequent end-to-end results'))
    result['input_options']=options
    # Use bounded parallel CPU prefetch as primary; device-prefetch separately benchmarked.
    result['primary_input_settings']=dict(parallel=4,prefetch=2,device_prefetch=False)
    r.write_json(args.output/'preprocessing.json',result)
    check_originals(args.output)
    return 0


def load_model(root,boundary):
    model,counts=r.build_model(); r.configure_stage(model,boundary)
    initial=root/'initial.weights.h5'
    if initial.exists():
        meta=json.loads((root/'initial.json').read_text()); r.sha256_file(initial,meta['sha256'])
        model.load_weights(initial)
        if r.weight_hashes(model.weights)!=meta['tensor_sha256']:
            raise ValueError('Model initialization differs from shared checkpoint')
    else:
        start=time.perf_counter(); model.save_weights(initial)
        r.write_json(root/'initial.json',dict(sha256=r.sha256_file(initial),tensor_sha256=r.weight_hashes(model.weights),
                                             save_seconds=time.perf_counter()-start,parameters=model.count_params(),counts=counts))
    return model,counts


def diagnostic_components(model,accumulator,images,labels):
    import tensorflow as tf
    forward=lambda x:model(x,training=False)
    forward(images).numpy(); totals=defaultdict(float)
    @tf.function
    def accumulate(gradients):
        for buffer,gradient in zip(accumulator.buffers,gradients):
            buffer.assign_add(gradient)
        return tf.reduce_sum(accumulator.buffers[-1])
    for repeat in range(4):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([getattr(v,'value',v) for v in model.trainable_variables])
            start=time.perf_counter(); probability=forward(images); probability.numpy(); f=time.perf_counter()-start
            start=time.perf_counter(); loss=tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels,probability)); loss.numpy(); l=time.perf_counter()-start
        start=time.perf_counter(); gradients=tape.gradient(loss,model.trainable_variables); tf.linalg.global_norm(gradients).numpy(); b=time.perf_counter()-start
        start=time.perf_counter(); accumulate(gradients).numpy(); a=time.perf_counter()-start
        if repeat:
            for key,value in [('forward',f),('loss',l),('backward',b),('gradient_accumulation',a)]:
                totals[key]+=value
    return dict(seconds=dict(totals),images=12,scope='Three diagnostic microbatches after one warmup; eager forward/tape in a fresh process without optimizer slots; backward includes global-norm synchronization; distinct from fused training timing')


def component_worker(args):
    root=args.output; r.logging_setup(root/args.job); configure('gpu')
    reader=c.CacheReader(root/'cache'); model,_=load_model(root,args.boundary)
    accumulator=r.Accumulator(model,r.adam(1e-5))
    images=np.stack([c.to_float(reader.get(i)) for i in range(4)])
    labels=np.array([x['ground_truth_code'] for x in reader.items[:4]],np.int64)
    result=compiled_components(model,accumulator,images,labels)
    r.write_json(root/args.job/'result.json',result); check_originals(root)
    return 0


def compiled_components(model,accumulator,images,labels):
    """Keep graph memory planning, with materialized CPU monotonic markers."""
    import tensorflow as tf
    def stamp(value):
        with tf.device('/CPU:0'):
            return tf.py_function(lambda completed:np.float64(time.perf_counter()),[value],tf.float64)
    @tf.function
    def measured(x,y):
        t0=stamp(tf.reduce_sum(x))
        with tf.GradientTape() as tape:
            with tf.control_dependencies([t0]):
                probability=model(x,training=False)
            t1=stamp(probability)
            with tf.control_dependencies([t1]):
                loss=tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y,probability))
            t2=stamp(loss)
        with tf.control_dependencies([t2]):
            gradients=tape.gradient(loss,model.trainable_variables)
            t3=stamp(tf.linalg.global_norm(gradients))
        with tf.control_dependencies([t3]):
            updates=[buffer.assign_add(gradient) for buffer,gradient in zip(accumulator.buffers,gradients)]
        with tf.control_dependencies(updates):
            t4=stamp(tf.constant(0))
        return tf.stack([t0,t1,t2,t3,t4])
    try:
        measured(images,labels).numpy()
        durations=np.stack([np.diff(measured(images,labels).numpy()) for _ in range(3)]).sum(axis=0)
        return dict(status='PASS',images=12,seconds=dict(zip(['forward','loss','backward','gradient_accumulation'],map(float,durations))),
            scope='Compiled graph with control-dependent CPU perf_counter markers; marker inputs materialize completed GPU values; backward includes gradient norm; diagnostic overhead differs from fused training')
    except tf.errors.ResourceExhaustedError as error:
        return dict(status='UNAVAILABLE_MEMORY',images=12,seconds=None,
                    reason=str(error),scope='Instrumentation exceeded memory; successful fused training timing remains authoritative')


def training_benchmark(args):
    import tensorflow as tf
    root=args.output; target=root/args.job; r.logging_setup(target)
    info=configure('gpu',asynchronous=args.mode in ('cached','device_prefetch'))
    reader=c.CacheReader(root/'cache'); rows=json.loads((root/'samples.json').read_text())
    model,counts=load_model(root,args.boundary); rate=next(lr for b,_,lr in r.SCHEDULE if b==args.boundary)
    accumulator=r.Accumulator(model,r.adam(rate)); reference=r.normalizers(r.REFERENCE) if args.mode=='online' else None
    frozen_before=r.weight_hashes(model.non_trainable_variables)
    inputs=None
    if args.mode=='pure':
        inputs=[]
        for offset in range(0,400,4):
            with tf.device('/GPU:0'):
                inputs.append(tf.stack([r.augment(c.to_float(reader.get(i)),rows[i]['sample_id'],1) for i in range(offset,offset+4)]))
        tf.reduce_sum(inputs[-1]).numpy()
    results=[]; preprocess_seconds=0.; iterator=None
    settings=dict(parallel=4,prefetch=2,device_prefetch=args.mode=='device_prefetch')
    if args.mode in ('cached','device_prefetch'):
        iterator=iter(c.dataset(reader,list(range(400)),**settings))
    for logical in range(4):
        with Utilization() as utilization:
            start=time.perf_counter(); micros=[]; load_times=[]; losses=[]
            for offset in range(logical*100,(logical+1)*100,4):
                begin=time.perf_counter()
                labels=np.array([rows[i]['ground_truth_code'] for i in range(offset,offset+4)],np.int64)
                if args.mode=='online':
                    images=tf.stack([r.augment(r.preprocess_legacy(rows[i],r.image_path(rows[i],args.extracted),reference),rows[i]['sample_id'],1) for i in range(offset,offset+4)])
                elif args.mode=='pure':
                    images=inputs[offset//4]
                else:
                    images,labels,ids=next(iterator)
                    if ids.numpy().tolist()!=list(range(offset,offset+4)):
                        raise ValueError('Benchmark loader ordering differs')
                load_times.append(time.perf_counter()-begin)
                begin=time.perf_counter(); losses.append(accumulator.add(images,labels)); micros.append(time.perf_counter()-begin)
            begin=time.perf_counter(); accumulator.flush(); update=time.perf_counter()-begin
            seconds=time.perf_counter()-start
        results.append(dict(seconds=seconds,images=100,images_per_second=100/seconds,input_seconds=sum(load_times),micro_seconds=micros,
                            optimizer_update_seconds=update,loss=float(np.mean(losses)),utilization=utilization.report()))
        r.LOG.info('%s %s logical %d/4: %.2f images/s',args.boundary,args.mode,logical+1,100/seconds)
    if frozen_before!=r.weight_hashes(model.non_trainable_variables):
        raise ValueError('Frozen weights changed')
    report=dict(boundary=args.boundary,mode=args.mode,device=info,trainable_parameters=counts[args.boundary],
                warmup=results[0],measured=results[1:],images_per_second=300/sum(x['seconds'] for x in results[1:]),
                seconds=sum(x['seconds'] for x in results[1:]),status='PASS',frozen_weights_stable=True)
    # Compiled validation versus eager; no held-out images, no updates.
    predict=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec((None,350,350,3),tf.float32)])
    sample=tf.convert_to_tensor(np.stack([c.to_float(reader.get(i)) for i in range(4)]))
    compiled=predict(sample).numpy(); eager=model(sample,training=False).numpy()
    report['compiled_eager_inference_parity']=dict(max_absolute=float(np.max(np.abs(compiled-eager))),
        pass_=bool(np.allclose(compiled,eager,atol=p.TOLERANCES['forward_probability_atol'],rtol=p.TOLERANCES['forward_probability_rtol'])))
    if not report['compiled_eager_inference_parity']['pass_']:
        raise ValueError('Compiled inference parity failed')
    validation=[]
    for count in range(3):
        with Utilization() as utilization:
            start=time.perf_counter()
            ds=c.dataset(reader,list(range(100)),augment=False,**settings)
            for x,y,ids in ds:
                predict(x).numpy()
            seconds=time.perf_counter()-start
        validation.append(dict(seconds=seconds,images_per_second=100/seconds,utilization=utilization.report()))
    report['cached_compiled_validation']=validation
    # Actual portable inference checkpoint I/O measurement, no production artifact.
    if args.mode=='cached' and args.boundary==r.SCHEDULE[0][0]:
        checkpoint=target/'disposable.keras'; start=time.perf_counter(); model.save(checkpoint)
        report['checkpoint_io']=dict(seconds=time.perf_counter()-start,bytes=checkpoint.stat().st_size,sha256=r.sha256_file(checkpoint))
        checkpoint.unlink()
    r.write_json(target/'result.json',report); check_originals(root)
    return 0


def save_vectors(path,variables,divide=1.):
    specs=[]; offset=0
    for index,var in enumerate(variables):
        n=int(np.prod(var.shape)); specs.append(dict(index=index,name=getattr(var,'path',var.name),shape=list(var.shape),offset=offset,elements=n)); offset+=n
    mapped=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(offset,))
    for var,spec in zip(variables,specs):
        mapped[spec['offset']:spec['offset']+spec['elements']]=var.numpy().reshape(-1)/np.float32(divide)
    mapped.flush(); del mapped
    return specs


def parity_worker(args):
    import tensorflow as tf
    root=args.output; target=root/args.job; r.logging_setup(target); info=configure(args.device)
    reader=c.CacheReader(root/'cache'); model,_=load_model(root,r.SCHEDULE[0][0])
    accumulator=r.Accumulator(model,r.adam(1e-5))
    x=np.stack([c.to_float(reader.get(i)) for i in range(100)]); y=np.array([row['ground_truth_code'] for row in reader.items[:100]],np.int64)
    predictor=tf.function(lambda z:model(z,training=False),input_signature=[tf.TensorSpec((None,350,350,3),tf.float32)])
    feature_model=tf.keras.Model(model.input,model.layers[-1].input)
    feature_predictor=tf.function(lambda z:feature_model(z,training=False),input_signature=[tf.TensorSpec((None,350,350,3),tf.float32)])
    def probe():
        probabilities=np.concatenate([predictor(x[i:i+4]).numpy() for i in range(0,16,4)])
        logits=[]
        for offset in range(0,16,4):
            features=feature_predictor(x[offset:offset+4])
            logits.append((tf.matmul(features,model.layers[-1].kernel)+model.layers[-1].bias).numpy())
        logits=np.concatenate(logits)
        loss=float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(y[:16],probabilities).numpy()))
        return dict(probabilities=probabilities.tolist(),logits=logits.tolist(),loss=loss)
    report=dict(device=info,initial_checkpoint_sha256=r.sha256_file(root/'initial.weights.h5'),before=probe(),updates=[],
                physical_batch=4,effective_batch=100,augmentation=False,labels=y.tolist(),
                inputs_sha256=hashlib.sha256(x.tobytes()).hexdigest(),tolerances=p.TOLERANCES)
    frozen=r.weight_hashes(model.non_trainable_variables)
    limit=3
    if args.device=='gpu' and (root/'parity_cpu'/'result.json').exists():
        limit=len(json.loads((root/'parity_cpu'/'result.json').read_text())['updates'])
    for update in range(1,limit+1):
        start=time.perf_counter(); losses=[]
        for offset in range(0,100,4):
            losses.append(accumulator.add(x[offset:offset+4],y[offset:offset+4]))
            if offset%20==0:
                r.LOG.info('%s parity update=%d images=%d/100',args.device,update,offset+4)
        train_seconds=time.perf_counter()-start
        if update==1:
            report['gradient_specs']=save_vectors(target/'gradients.npy',accumulator.buffers,100)
        accumulator.flush()
        specs=save_vectors(target/f'weights_{update}.npy',model.weights)
        report['weight_specs']=specs
        report['updates'].append(dict(update=update,pre_update_loss=float(np.mean(losses)),training_seconds=train_seconds,after=probe()))
        r.write_json(target/'result.json',report)
        if args.device=='cpu' and update==1 and train_seconds>900:
            report['multi_update_skipped_reason']='First CPU logical batch exceeded predeclared 900-second practicality limit'
            break
    if frozen!=r.weight_hashes(model.non_trainable_variables):
        raise ValueError('Parity run updated frozen weights')
    report.update(status='COMPLETE',frozen_weights_stable=True)
    r.write_json(target/'result.json',report); check_originals(root)
    return 0


def compare_parity(root):
    cpu=json.loads((root/'parity_cpu'/'result.json').read_text()); gpu=json.loads((root/'parity_gpu'/'result.json').read_text())
    if cpu['initial_checkpoint_sha256']!=gpu['initial_checkpoint_sha256'] or cpu['inputs_sha256']!=gpu['inputs_sha256'] or cpu['labels']!=gpu['labels']:
        raise ValueError('CPU/GPU starting state or input mismatch')
    result=dict(before=p.forward_comparison(cpu['before'],gpu['before']),gradients=[],updates=[],tolerances=p.TOLERANCES,rationale=p.RATIONALE)
    a=np.load(root/'parity_cpu'/'gradients.npy',mmap_mode='r'); b=np.load(root/'parity_gpu'/'gradients.npy',mmap_mode='r')
    if cpu['gradient_specs']!=gpu['gradient_specs']:
        raise ValueError('Gradient variable ordering mismatch')
    for spec in cpu['gradient_specs']:
        offset=spec['offset']; end=offset+spec['elements']; stats=p.difference(a[offset:end],b[offset:end])
        result['gradients'].append(dict(spec=spec,stats=stats,pass_=p.gradient_pass(stats)))
    for cu,gu in zip(cpu['updates'],gpu['updates']):
        update=cu['update']; a=np.load(root/'parity_cpu'/f'weights_{update}.npy',mmap_mode='r'); b=np.load(root/'parity_gpu'/f'weights_{update}.npy',mmap_mode='r')
        stats=p.difference(a,b); forward=p.forward_comparison(cu['after'],gu['after'],post=True)
        result['updates'].append(dict(update=update,weights=stats,weights_pass=p.weights_pass(stats,update),after=forward,
                                      pre_update_loss_cpu=cu['pre_update_loss'],pre_update_loss_gpu=gu['pre_update_loss'],
                                      pre_update_loss_difference=abs(cu['pre_update_loss']-gu['pre_update_loss'])))
    result['pass']=bool(result['before']['pass'] and all(x['pass_'] for x in result['gradients']) and
                        all(x['weights_pass'] and x['after']['pass'] and x['pre_update_loss_difference']<=p.TOLERANCES['post_loss_atol'] for x in result['updates']))
    result['cpu_runtime_seconds']=[x['training_seconds'] for x in cpu['updates']]
    r.write_json(root/'parity_comparison.json',result)
    return result


def launch(args,job,**options):
    if args.resume and (args.output/job/'result.json').exists():
        saved=json.loads((args.output/job/'result.json').read_text())
        if not job.startswith('parity_') or saved.get('status')=='COMPLETE':
            r.LOG.info('Preserving completed job %s',job)
            return
    if (args.output/job).exists():
        prior=args.output/job; archived=args.output/(job+'_failed_'+str(time.time_ns()))
        if prior.resolve().parent!=args.output.resolve() or archived.resolve().parent!=args.output.resolve():
            raise ValueError('Unsafe failed-attempt archive path')
        prior.rename(archived)
        console=args.output/(job+'.console.log')
        if console.exists(): console.rename(args.output/(archived.name+'.console.log'))
    command=[sys.executable,'-B','-m','modern_pca.profile_reimplementation','--worker',job,'--output',str(args.output),'--extracted',str(args.extracted)]
    for key,value in options.items():
        command.extend(['--'+key.replace('_','-'),str(value)])
    with (args.output/(job+'.console.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        while process.poll() is None:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: r.LOG.info('Still running %s; see console log',job)
    if process.returncode:
        raise RuntimeError(f'{job} failed ({process.returncode}); see {job}.console.log')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--extracted',type=Path,required=True)
    parser.add_argument('--worker'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--mode',choices=['online','cached','pure','device_prefetch'],default='cached')
    parser.add_argument('--boundary',choices=[x[0] for x in r.SCHEDULE],default=r.SCHEDULE[0][0])
    parser.add_argument('--device',choices=['cpu','gpu'],default='gpu'); args=parser.parse_args(argv)
    r.ensure_output_separation(args.extracted,[args.output])
    if args.worker:
        args.job=args.worker
        if args.worker=='prepare': return prepare(args)
        if args.worker=='input':
            r.logging_setup(args.output/'input'); configure('gpu'); return input_benchmarks(args)
        if args.worker.endswith('_components'): return component_worker(args)
        if args.worker.startswith('parity_'): return parity_worker(args)
        return training_benchmark(args)
    if args.resume:
        check_originals(args.output)
        resume_name='resume_'+str(time.time_ns())
        r.logging_setup(args.output/resume_name)
        r.write_json(args.output/(resume_name+'_implementation.json'),dict(timestamp=r.utc_now(),
            reason='Resume incomplete bounded jobs; compiled component diagnostics and physical-batch-four logit probes bound diagnostic memory; scientific settings and tolerances unchanged',
            implementation_sha256=r.sha256_file(Path(__file__))))
        if 'input_options' not in json.loads((args.output/'preprocessing.json').read_text()):
            launch(args,'input')
    else:
        initialize_run(args)
        launch(args,'prepare')
    for boundary,_,_ in r.SCHEDULE:
        launch(args,f'{boundary}_cached',boundary=boundary)
    for boundary in (r.SCHEDULE[0][0],r.SCHEDULE[-1][0]):
        for mode in ('online','pure','device_prefetch'):
            launch(args,f'{boundary}_{mode}',boundary=boundary,mode=mode)
        launch(args,f'{boundary}_components',boundary=boundary)
    launch(args,'parity_cpu',device='cpu')
    launch(args,'parity_gpu',device='gpu')
    compare_parity(args.output); check_originals(args.output)
    r.write_json(args.output/'run_complete.json',dict(completed_at_utc=r.utc_now(),status='COMPLETE',production_training=False,VAL2_accessed=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
