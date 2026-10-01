"""Build Stage 2E reports from completed, bounded benchmark artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pooled_utilization(repeats):
    trace=[s for x in repeats for s in x['utilization'].get('trace',[])]
    if not trace:
        return {}
    result={}
    for key in ('gpu_percent','cpu_system_percent','cpu_process_percent_one_core'):
        values=[s[key] for s in trace if s.get(key) is not None]
        result[key]=sum(values)/len(values) if values else None
    result['peak_gpu_memory_mib']=max(s['gpu_memory_mib'] for s in trace)
    result['low_gpu_fraction']=sum(s['gpu_percent']<10 for s in trace)/len(trace)
    return result


def estimate(stages,cache_seconds_per_image,checkpoint_seconds,startup_seconds=0.):
    """Planning model: measured steady state, explicit non-statistical margins."""
    rows=[]
    for stage in stages:
        training=116655/stage['train_images_per_second']
        validation=29164/stage['validation_images_per_second']
        rows.append(dict(stage,training_seconds_per_epoch=training,validation_seconds_per_epoch=validation,
                         total_seconds_per_epoch=training+validation))
    train=sum(x['epochs']*x['training_seconds_per_epoch'] for x in rows)
    val=sum(x['epochs']*x['validation_seconds_per_epoch'] for x in rows)
    cache=145819*cache_seconds_per_image
    # WSL/NTFS cold shuffled throughput was not measured on a full 53.6 GB cache.
    # Allow a conservative, explicitly assumed 20 MiB/s input floor, overlapped
    # with compute by prefetch. This is a sensitivity calculation, not a CI.
    input_per_epoch=145819*367500/(20*1024**2)
    upper_compute=sum(x['epochs']*max(1.30*x['total_seconds_per_epoch'],input_per_epoch) for x in rows)
    return dict(stages=rows,cache_generation_seconds=cache,training_seconds=train,validation_seconds=val,
                checkpoint_io_seconds=checkpoint_seconds,startup_compilation_seconds=startup_seconds,
                epochs_14_seconds=train+val+checkpoint_seconds+startup_seconds,
                expected_complete_seconds=cache+train+val+checkpoint_seconds+startup_seconds,
                conservative_complete_seconds=[cache+train+val+checkpoint_seconds+startup_seconds,
                                               1.30*cache+upper_compute+2*(checkpoint_seconds+startup_seconds)+input_per_epoch],
                assumptions=['Steady state extrapolation from 300 measured training samples per stage after 100 warmup samples.',
                             'Validation extrapolated from training images; no internal-validation or VAL2 pixels accessed.',
                             'One final portable checkpoint, matching the existing production script; checkpoint timing excludes checksum verification.',
                             'Upper planning bound: 30% cache/compute slowdown, 20 MiB/s cold shuffled input floor, double checkpoint/startup cost and one full-cache verification read at that assumed rate.',
                             'Range is conditional, not a statistical confidence interval or guaranteed thermal/disk bound.',
                             'Model load measured separately; startup estimate adds excess first-logical-batch warmup for each stage and compiled-validation warmup per stage.',
                             'Full-cache post-generation verification and metadata startup costs remain unmeasured; allow operational slack beyond this conditional estimate.'])


def build(root):
    protocol=read(root/'protocol.json'); prep=read(root/'preprocessing.json')
    complete=read(root/'run_complete.json'); parity=read(root/'parity_comparison.json')
    cpu=read(root/'parity_cpu'/'result.json')
    weights=cpu['weight_specs']; start=next(i for i,x in enumerate(weights) if '/normal_conv_1_17/' in x['name'])
    trainable=[x for x in weights[start:] if x['name'].split('/')[-1] not in ('gamma','beta','moving_mean','moving_variance')]
    if len(trainable)!=len(parity['gradients']) or any(a['shape']!=b['spec']['shape'] for a,b in zip(trainable,parity['gradients'])):
        raise ValueError('Saved gradient/weight ordering cannot be mapped safely')
    t=parity['tolerances']
    for weight,gradient in zip(trainable,parity['gradients']):
        gradient['layer_name']=weight['name']; z=gradient['stats']; failures=[]
        if z['max_absolute']>t['gradient_max_atol']+t['gradient_max_rtol']*z['reference_max_absolute']:
            failures.append('maximum absolute / tensor-scale bound')
        if z['mean_absolute']>t['gradient_mean_atol']+t['gradient_mean_rtol']*z['reference_mean_absolute']:
            failures.append('mean absolute / tensor-scale bound')
        if z['relative_l2']>t['gradient_relative_l2'] and z['reference_l2']>=1e-6:
            failures.append('relative L2 bound')
        gradient['failed_criteria']=failures
    validation=read(root/'validation_comparison.json'); samples=read(root/'samples.json')
    jobs={path.parent.name:read(path) for path in sorted(root.glob('normal_conv*/result.json'))
          if not path.parent.name.endswith('_components') and '_failed_' not in path.parent.name}
    for boundary in (protocol['stages'][0][0],protocol['stages'][-1][0]):
        jobs[boundary+'_pure']['components']=read(root/(boundary+'_components')/'result.json')
    jobs[protocol['stages'][0][0]+'_pure']['components']=read(root/'first_stage_compiled_components.json')
    expected={f'{b}_cached' for b,_,_ in protocol['stages']}
    expected|={f'{b}_{m}' for b in (protocol['stages'][0][0],protocol['stages'][-1][0])
               for m in ('online','pure','device_prefetch')}
    if set(jobs)!=expected or complete['status']!='COMPLETE':
        raise ValueError('Incomplete benchmark artifacts')
    for filename,digest in protocol['protected_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Protected artifact changed: {filename}')
    meta=prep['cache_generation']; timings=meta['components_seconds']; wall=meta['generation_seconds']; n=meta['samples']
    metadata_bytes=sum(path.stat().st_size for path in (root/'cache').glob('*.json'))
    prep['full_VAL1_storage']['subset_json_metadata_bytes']=metadata_bytes
    prep['full_VAL1_storage']['projected_json_metadata_bytes']=round(metadata_bytes/n*145819)
    prep['full_VAL1_storage']['metadata_estimate_note']='Linear projection of subset JSON including duplicated item sidecars/global index; future full-cache schema may differ.'
    post=sum(timings[k] for k in ('float32_conversion','scaling','augmentation','batching','host_to_GPU_transfer'))
    components={key:dict(seconds=value,seconds_per_image=value/n,images_per_second=n/value if value else None,
                         percent_instrumented_wall=100*value/wall) for key,value in timings.items()}
    residual=wall-sum(timings.values())
    cache_seconds_per_image=(wall-post)/n
    stages=[]
    for boundary,epochs,lr in protocol['stages']:
        job=jobs[boundary+'_cached']
        stages.append(dict(boundary=boundary,epochs=epochs,learning_rate=lr,train_images_per_second=job['images_per_second'],
                           validation_images_per_second=300/sum(x['seconds'] for x in job['cached_compiled_validation'])))
    checkpoint=jobs[protocol['stages'][0][0]+'_cached']['checkpoint_io']
    startup=validation['model_load_seconds']+8*validation['compiled_warmup_seconds']
    startup+=sum(max(0,jobs[x['boundary']+'_cached']['warmup']['seconds']-100/x['train_images_per_second']) for x in stages)
    forecast=estimate(stages,cache_seconds_per_image,checkpoint['seconds'],startup)
    cache_pass=all(x['uint8']['pass_'] and x['float32']['pass_'] for x in prep['parity'])
    formats_pass=all(v['exact_parity'] for k,v in prep['formats'].items() if k!='write_seconds')
    input_pass=all(x['exact_parity'] for x in prep['input_options'])
    reasons=['Full-cache generation remains unapproved and unperformed.',
             'Cache loader is a bounded profiling implementation, not yet integrated into the production trainer.',
             'Full-cohort cache identity, complete coverage and cold shuffled I/O must be verified before launch.']
    if not parity['pass']:
        reasons.append('CPU/GPU parity failed predeclared acceptance criteria; investigate before production.')
    summary=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),stage='2E',run=str(root),
        scientific_scope='Constrained binary reimplementation, bounded VAL1-training diagnostics only',
        original_stage2d_hours=[166.7,177.0],cache_parity_pass=cache_pass,formats_parity_pass=formats_pass,
        input_optimization_parity_pass=input_pass,cpu_gpu_parity_pass=parity['pass'],
        subset=dict(samples=n,classes=dict(Counter(x['class_name'] for x in samples)),
                    dimensions=dict(Counter(f"{x['width']}x{x['height']}" for x in samples))),
        preprocessing=dict(components=components,instrumented_wall_seconds=wall,
                           unattributed_overhead_seconds=residual,
                           serial_A_to_E_images_per_second=n/sum(timings[k] for k in ('file_read','decode_rgb','resize','brightness','macenko')),
                           macenko_percent_A_to_E=100*timings['macenko']/sum(timings[k] for k in ('file_read','decode_rgb','resize','brightness','macenko')),
                           online_input_images_per_second={b:300/sum(x['input_seconds'] for x in jobs[b+'_online']['measured'])
                                                           for b in (protocol['stages'][0][0],protocol['stages'][-1][0])},
                           estimated_cache_generation_images_per_second=1/cache_seconds_per_image,
                           scope='Serial pass with cache writing and F-J instrumentation. Cache estimate subtracts F-J; unattributed residual includes path/stat safety checks, shard writes, hashes, allocation and logging.'),
        recommended_format='256-image uncompressed uint8 NPY shards, mmap read-only, per-item and per-shard SHA-256',
        full_VAL1_storage=prep['full_VAL1_storage'],forecast=forecast,
        go_full_cache_generation='GO (recommendation only)' if cache_pass and formats_pass and input_pass else 'NO-GO',
        go_production_training='NO-GO',production_reasons=reasons,
        benchmark_utilization={key:pooled_utilization(value['measured']) for key,value in jobs.items()},
        protocol=protocol,preparation=prep,training_benchmarks=jobs,validation_comparison=validation,
        resume_history={path.name:read(path) for path in sorted(root.glob('resume*implementation.json'))},
        reporting_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_gpu_comparison=parity,completion=complete,
        restrictions_verified=dict(VAL2_images_accessed=False,source_JPEGs_modified=False,full_cache_generated=False,
                                   production_training_started=False,protected_artifacts_sha256_pass=True),
        limitations=['1000-image format benchmark fits OS cache; warm mmap rates are not cold full-cache disk performance.',
                     'Execution spans October 1-2 after a profiler failure/resume; laptop thermal/power variation is not controlled between jobs.',
                     'Training timings use the first 400 interleaved subset images (200 per class); the complete 1000-image profiling subset follows the training class ratio.',
                     'Validation timings include construction/iteration of small 100-image datasets; fixed setup overhead makes per-image extrapolation conservative.',
                     'GPU utilization <10% is a sampled idle proxy, not measured kernel stall time; nvidia-smi includes other desktop processes.',
                     'CPU system percent spans all logical CPUs; process percent uses one-core=100%.',
                     'GPU memory from nvidia-smi includes allocator reservations and other processes, not only live model tensors.',
                     'Diagnostic forward/backward phase timers use a different execution graph from fused training; do not sum them to predict epochs.',
                     'CPU/GPU parity uses Linux CPU and GPU in the same WSL environment, not a Windows/WSL environment-equivalence claim.',
                     'Short-update parity does not establish equivalent long training trajectories.',
                     'The benchmark uses deterministic training subsets; it does not measure predictive generalization.'])
    return summary


def render(s):
    prep=s['preparation']; jobs=s['training_benchmarks']; forecast=s['forecast']; parity=s['cpu_gpu_comparison']
    lines=['# Stage 2E: bounded performance and numerical-parity report','',
           f"Generated {s['created_at_utc']}. Raw evidence: `{s['run']}`.",'',
           f"**Cache parity: {'PASS' if s['cache_parity_pass'] else 'FAIL'}. CPU/GPU parity: {'PASS' if s['cpu_gpu_parity_pass'] else 'FAIL'}.**",
           f"Full-cache recommendation: **{s['go_full_cache_generation']}**. Production launch: **NO-GO**.",'',
           'No production training or full cache was started. No VAL2 images were accessed. Source JPEGs, strict original-checkpoint evaluator, model/training configuration and frozen split were not changed.',
           '', '## Scope and controls','',
           f"1000 deterministic VAL1-training samples: {s['subset']['classes']}; native dimensions: {s['subset']['dimensions']}. Exact cache parity re-ran the unchanged legacy helper on 64 representative samples, including both labels and all selected dimension strata. All 1000 source hashes were rechecked. Four formats and every optimized loader configuration preserved every tested value.",
           '', 'All training probes use the same saved initialization, ImageNet weights, 350x350 input, 209,812,820 parameters, float32 with TF32 disabled, frozen BN, physical batch 4 and effective batch 100. One logical batch warms up; three are measured. These disposable short runs are profiling, not fitted models for scientific evaluation.',
           '', '## Preprocessing','',
           '| Component | Total s | ms/image | images/s | % instrumented wall |','|---|---:|---:|---:|---:|']
    for name,x in s['preprocessing']['components'].items():
        lines.append(f"| {name} | {x['seconds']:.4f} | {1000*x['seconds_per_image']:.4f} | {x['images_per_second']:.2f} | {x['percent_instrumented_wall']:.2f} |")
    x=s['preprocessing']
    lines.extend(['',f"Instrumented wall: {x['instrumented_wall_seconds']:.2f} s. Unattributed path/stat safety checks, cache writing, hashing, allocation and logging overhead: {x['unattributed_overhead_seconds']:.2f} s. These residual costs were not separated and are not assigned to Macenko. A-E operations alone: {x['serial_A_to_E_images_per_second']:.3f} images/s; cache-generation estimate including observed residual: {x['estimated_cache_generation_images_per_second']:.3f} images/s. F-J are excluded from the one-time cache cost because the stored tensor precedes those operations.",
        '', 'Macenko is the largest measured preprocessing operation and runs on CPU. Compare endpoint online/cached GPU activity below to assess input starvation. The instrumented serial pass also includes small H2D probes; its utilization is not a Macenko-only measurement.',
        '', f"Macenko accounts for {x['macenko_percent_A_to_E']:.2f}% of timed A-E preprocessing operations. Separately, online input preparation during training (including source-path checks, scaling, flips and batching) measured {x['online_input_images_per_second']['normal_conv_1_17']:.2f} images/s at the first stage and {x['online_input_images_per_second']['normal_conv_1_1']:.2f} images/s at the deepest stage; these rates exclude the subsequent training microbatch.",
        '', f"Serial preprocessing utilization: `{json.dumps({k:v for k,v in prep['serial_preprocessing_utilization'].items() if k!='trace'})}`.",
        '', 'All timings use `time.perf_counter()`. GPU tensors and compiled outputs are materialized with `.numpy()` before stopping timers; H2D includes a scalar reduction and synchronization. Backward diagnostics materialize a gradient norm. Timings therefore include completion, not just GPU launch latency.',
        '', '## Cache formats','',
        '| Format | Subset bytes | Sequential images/s | Shuffled images/s | Exact |',
        '|---|---:|---:|---:|---|'])
    for name,v in prep['formats'].items():
        if name!='write_seconds':
            lines.append(f"| {name} | {v['bytes']:,} | {v['sequential']['images_per_second']:.1f} | {v['shuffled']['images_per_second']:.1f} | {v['exact_parity']} |")
    lines.extend(['', 'Read rates are medians of three warm-filesystem-cache passes copying full arrays; they are not full-dataset cold-disk rates. TFRecord sequential and shuffled figures use the same indexed raw-record reader; TensorFlow native CRC-checked decoding was separately round-trip verified. They do not benchmark an optimized streaming TFRecord pipeline.',
        '', '**Recommend 256-image NPY shards.** They retain shape/dtype headers, allow direct read-only mmap and random sample access, and avoid 145,819 small-file opens. Individual NPY is simple to resume but costly on the WSL/NTFS boundary. Raw mmap is similarly fast but requires an external schema and offsets. TFRecord has native TensorFlow streaming support and CRCs, but efficient arbitrary shuffled access requires an index or changes to loading order; the indexed prototype is slower here.',
        '', 'Use one writer, temporary shard + fsync + atomic rename, then a committed SHA-256 sidecar. Resume only matching context and verified committed shards. Each process constructs its own read-only mmap handles; parallel reads do not mutate shared arrays. Per-image source and tensor hashes permit individual verification. Production conversion needs a train/validation-aware extension; the present builder deliberately caps at 1024 training samples.',
        '', 'Metadata records sample ID, relative source path, JPEG/tensor SHA-256, shape, dtype, class, split, shard and index. Global context records source/split/reference hashes, implementation fingerprints, resize/brightness/Macenko settings, dependency versions and backend identities. Augmentation is never cached.',
        '', f"Full VAL1 tensor payload: **{s['full_VAL1_storage']['raw_tensor_bytes']:,} bytes**. Separate train/validation 256-image NPY shards: **{s['full_VAL1_storage']['npy_bytes']:,} bytes** ({s['full_VAL1_storage']['npy_bytes']/1e9:.3f} GB; {s['full_VAL1_storage']['npy_bytes']/1024**3:.3f} GiB), {s['full_VAL1_storage']['shards']} shards. Exact container total excludes JSON metadata, filesystem allocation and temporary workspace. Preserve source JPEGs separately; budget additional space for hashes/metadata and one temporary shard.",
        '', f"Subset JSON metadata occupies {s['full_VAL1_storage']['subset_json_metadata_bytes']:,} bytes. A linear projection is approximately {s['full_VAL1_storage']['projected_json_metadata_bytes']/1e6:.1f} MB for full VAL1; this is an estimate, not an exact future metadata-file size.",
        '', '## Input optimization checks','',
        '| Parallel readers | Prefetch | Device prefetch | images/s including verification | Exact values/order/labels/flips |',
        '|---:|---:|---|---:|---|'])
    for opt in prep['input_options']:
        a=opt['settings']; lines.append(f"| {a['parallel']} | {a['prefetch']} | {a['device_prefetch']} | {opt['images_per_second']:.2f} | {opt['exact_parity']} |")
    lines.extend(['', 'Primary forecast uses four bounded parallel readers and two host-prefetched batches. Device-prefetch results are measured separately below. No explicit pinned-memory allocation is claimed: TensorFlow controls transfers, and `prefetch_to_device` is the tested overlap mechanism. NumPy float32 division is preserved inside the pure cache-read callback to avoid graph rewrite to rounded reciprocal multiplication. Seeded sample/epoch stateless flips and deterministic order match the existing trainer.',
        '', '## Training and utilization','',
        '| Stage / pipeline | measured s / 300 | training img/s | mean GPU % | mean CPU system % | process CPU % (100=one core) | peak GPU MiB | GPU <10% fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name,job in jobs.items():
        u=s['benchmark_utilization'][name]
        lines.append(f"| {name} | {job['seconds']:.2f} | {job['images_per_second']:.2f} | {u.get('gpu_percent',0):.1f} | {u.get('cpu_system_percent',0):.1f} | {u.get('cpu_process_percent_one_core',0):.1f} | {u.get('peak_gpu_memory_mib',0):.0f} | {u.get('low_gpu_fraction',0):.3f} |")
    lines.extend(['', 'Utilization is sampled at approximately 0.5-second intervals; raw traces are retained. Low utilization is an idle proxy, not an exact stall percentage. Pure mode preloads float32 tensors onto GPU; online mode repeats strict JPEG preprocessing. Online/pure use synchronous execution, matching Stage 2D; cached/device-prefetch permit asynchronous dispatch with materialization at each completed microbatch/update. Therefore the speedup measures the safe optimization bundle, not an isolated cache-only causal effect.',
        '', '### GPU computation components','',
        '| Stage | Forward s | Loss s | Backward s | Accumulation s | Mean Adam update s |',
        '|---|---:|---:|---:|---:|---:|'])
    for job in jobs.values():
        if 'components' in job:
            z=job['components']['seconds']; adam=sum(x['optimizer_update_seconds'] for x in job['measured'])/3
            if z is None:
                lines.append(f"| {job['boundary']} | N/A (instrumentation OOM) | N/A | N/A | N/A | {adam:.6f} |")
            else:
                lines.append(f"| {job['boundary']} | {z['forward']:.6f} | {z['loss']:.6f} | {z['backward']:.6f} | {z['gradient_accumulation']:.6f} | {adam:.6f} |")
    for job in jobs.values():
        if 'components' in job:
            lines.extend(['',f"Scope ({job['boundary']}): {job['components']['scope']}"])
    lines.extend(['', 'First four columns total three diagnostic microbatches (12 images), after warmup. Adam is per logical 100-image update from fused training. Compiled phase diagnostics have materialized CPU timing markers and no allocated Adam slots. They include instrumentation/synchronization overhead and must not be added to predict ordinary fused training speed. An earlier eager first-stage probe is retained separately in the run artifacts.',
        '', '## Separate validation inference','',
        '| Pipeline | images/s | mean GPU % | peak GPU MiB |', '|---|---:|---:|---:|'])
    for name,v in s['validation_comparison']['results'].items():
        u=pooled_utilization(v['repeats'])
        lines.append(f"| {name} | {v['images_per_second']:.2f} | {u.get('gpu_percent',0):.1f} | {u.get('peak_gpu_memory_mib',0):.0f} |")
    lines.extend(['', 'Same 100 training images, three repeats, unchanged initial checkpoint, no augmentation/updates. Eager vs compiled and online vs cache predictions must satisfy predeclared forward tolerances and exact class agreement. Internal-validation images were not read; these are timing surrogates, not validation accuracy results.',
        '', '## Revised runtime forecast','', '| Epoch(s) | Unfreeze boundary | Training img/s | Validation img/s | Training h/epoch | Validation h/epoch | Total h/epoch |', '|---|---|---:|---:|---:|---:|---:|'])
    epoch=1
    for stage in forecast['stages']:
        label=f"{epoch}-{epoch+stage['epochs']-1}" if stage['epochs']>1 else str(epoch); epoch+=stage['epochs']
        lines.append(f"| {label} | {stage['boundary']} | {stage['train_images_per_second']:.2f} | {stage['validation_images_per_second']:.2f} | {stage['training_seconds_per_epoch']/3600:.3f} | {stage['validation_seconds_per_epoch']/3600:.3f} | {stage['total_seconds_per_epoch']/3600:.3f} |")
    lines.extend(['',f"Stage 2D serial estimate: **166.7-177.0 hours**. Stage 2E: one-time cache **{forecast['cache_generation_seconds']/3600:.2f} h**; training **{forecast['training_seconds']/3600:.2f} h**; validation **{forecast['validation_seconds']/3600:.2f} h**; final checkpoint **{forecast['checkpoint_io_seconds']:.2f} s**. Fourteen epochs including validation/checkpoint: **{forecast['epochs_14_seconds']/3600:.2f} h**. Complete expected experiment including cache: **{forecast['expected_complete_seconds']/3600:.2f} h**; conditional conservative planning range **{forecast['conservative_complete_seconds'][0]/3600:.2f}-{forecast['conservative_complete_seconds'][1]/3600:.2f} h**.",''])
    lines.append(f"Estimated model-loading/compilation allowance included in totals: {forecast['startup_compilation_seconds']:.1f} s.\n")
    lines.extend('- '+x for x in forecast['assumptions'])
    lines.extend(['', '## CPU/GPU numerical parity','', 'Acceptance criteria were saved in `protocol.json` before measurements. No tolerance was relaxed after observing results.', '', '```json',json.dumps(parity['tolerances'],indent=2),'```','',parity['rationale'],'',
        f"Shared checkpoint SHA-256: `{read(Path(s['run'])/'initial.json')['sha256']}`. Identical 100 cached images/labels, physical batch 4, fresh identical Adam, no augmentation. The CPU and GPU run in the same WSL package environment. Forward/post-update prediction probes use 16 of those images; training gradients use all 100.",
        '',f"Forward PASS: {parity['before']['pass']}; probability statistics: `{json.dumps(parity['before']['probabilities'])}`; logits: `{json.dumps(parity['before']['logits'])}`; loss difference {parity['before']['loss_difference']:.8g}; class agreement {parity['before']['predicted_class_agreement']:.6f}.",
        '', f"Gradients: {sum(x['pass_'] for x in parity['gradients'])}/{len(parity['gradients'])} tensors passed. Maximum absolute difference across tensors: {max(x['stats']['max_absolute'] for x in parity['gradients']):.8g}. Per-tensor max/mean/relative/L2 statistics and shapes are retained in JSON; buffer index follows model trainable-variable order.",
        '', '| Update | CPU pre-loss | GPU pre-loss | Weight max abs | Weight mean abs | Post probability max abs | Post class agreement | Pass |',
        '|---:|---:|---:|---:|---:|---:|---:|---|'])
    for x in parity['updates']:
        lines.append(f"| {x['update']} | {x['pre_update_loss_cpu']:.8f} | {x['pre_update_loss_gpu']:.8f} | {x['weights']['max_absolute']:.8g} | {x['weights']['mean_absolute']:.8g} | {x['after']['probabilities']['max_absolute']:.8g} | {x['after']['predicted_class_agreement']:.6f} | {x['weights_pass'] and x['after']['pass'] and x['pre_update_loss_difference']<=parity['tolerances']['post_loss_atol']} |")
    failed=[x for x in parity['gradients'] if not x['pass_']]
    if failed:
        lines.extend(['','### Gradient failures (unchanged acceptance limits)','',
                      '| Variable | Max absolute difference | Mean absolute difference | Relative L2 | Failed criterion |',
                      '|---|---:|---:|---:|---|'])
        for x in failed:
            z=x['stats']; lines.append(f"| {x['layer_name']} | {z['max_absolute']:.8g} | {z['mean_absolute']:.8g} | {z['relative_l2']:.8g} | {', '.join(x['failed_criteria'])} |")
        lines.extend(['', 'Names are mapped from saved model-weight ordering after the first unfreeze boundary, excluding frozen BatchNorm variables; all 47 counts/shapes were verified. These are real failures of the registered criteria even though predictions and post-update weights pass.',
                      '', 'Next investigation: localize failing sample/layer gradients, verify repeatability within each device, and inspect near-zero activation boundaries and CPU/CUDA reduction differences. These are hypotheses, not established causes. Any alternative CPU backend or acceptance policy requires a separately identified follow-up; do not retroactively relax these limits.'])
    lines.extend(['',f"CPU logical-update seconds: {parity['cpu_runtime_seconds']}. Predeclared practicality rule: if the first CPU update exceeds 900 seconds, omit subsequent updates. Overall numerical parity: **{'PASS' if parity['pass'] else 'FAIL'}**. Individual failure details remain in JSON; raw logits/probabilities/losses are in `parity_cpu/result.json` and `parity_gpu/result.json` under the run directory. Short-run agreement never proves identical 14-epoch trajectories.",
        '', '## Recommendations and remaining gates',''])
    lines.extend('- '+x for x in s['production_reasons'])
    lines.extend(['', 'If approved later, generate and verify separate full train/internal-validation caches; integrate this verified loader into the separate reimplementation trainer; retain original shuffling, stateless flips, full sample coverage, final partial logical-batch handling and staged Adam resets. Measure cold shuffled reads and sustained laptop thermals before scheduling the full run. Keep VAL2 sealed.',
        '', '## Limitations',''])
    lines.extend('- '+x for x in s['limitations'])
    lines.extend(['', '## Reproducibility and checks','',
        'Profiling code: `modern_pca/preprocessing_cache.py`, `modern_pca/profile_reimplementation.py`, `modern_pca/profile_validation.py`, `modern_pca/numerical_parity.py`. Report builder: `tools/report_stage2e.py`. Tests: `tests/test_stage2e.py`.',
        '', 'The first preparation attempt completed cache parity, then stopped on an ndarray/tensor return-type handling error in the profiling check. The check was corrected and resumed from the verified cache. Original protocol and logs were retained; `resume_implementation.json` records the new profiling-source fingerprint. Scientific preprocessing and predeclared tolerances were unchanged.',
        '', 'A later phase-by-phase diagnostic exhausted GPU memory when its additional gradient graph coexisted with the trained model/Adam slots. Cached training at all eight stages had already succeeded. A fresh eager diagnostic passed at the first boundary but exhausted memory at the deepest boundary. Final component diagnostics use compiled graph memory planning with control-dependent, materialized CPU perf_counter markers. Failed logs are preserved; resume records bind profiling-source versions. Fused training configuration was unchanged.',
        '', 'The initial CPU logit-only probe used an eager batch of 16 and exhausted memory before any optimizer update. Final CPU/GPU logits are computed in compiled batches of four, matching the probability/training probe physical batch. Initialization, 100-image logical updates and acceptance tolerances were unchanged.',
        '', 'Protected artifacts are rehashed by the report builder. Test and Git check evidence is recorded separately in `reports/stage2e_checks.json`.',
        '', 'TensorFlow references: [input pipeline performance](https://www.tensorflow.org/guide/data_performance), [performance analysis](https://www.tensorflow.org/guide/data_performance_analysis), [prefetch_to_device](https://www.tensorflow.org/api_docs/python/tf/data/experimental/prefetch_to_device).',''])
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage2e_profile'))
    parser.add_argument('--reports',type=Path,default=Path('reports'))
    parser.add_argument('--complete-run',action='store_true',help='After completed parity, run remaining validation timings and CPU unit tests before reporting')
    args=parser.parse_args(argv); args.reports.mkdir(exist_ok=True)
    if args.complete_run:
        if read(args.run/'run_complete.json')['status']!='COMPLETE':
            raise ValueError('Main bounded benchmarks must finish before validation/tests')
        if not (args.run/'validation_comparison.json').exists():
            protocol=read(args.run/'protocol.json')
            command=[sys.executable,'-B','-m','modern_pca.profile_validation','--output',str(args.run),'--extracted',protocol['extracted']]
            with (args.run/'validation_comparison.console.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider']
        with (args.reports/'stage2e_tests.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1'))
        checks=dict(timestamp=datetime.now(timezone.utc).isoformat(),unit_test_command=command,
                    unit_tests_returncode=result.returncode,unit_test_log='reports/stage2e_tests.log',
                    git_checks='Pending final Windows worktree checks')
        (args.reports/'stage2e_checks.json').write_bytes((json.dumps(checks,indent=2)+'\n').encode())
        if result.returncode:
            raise RuntimeError('Unit tests failed; see reports/stage2e_tests.log')
    summary=build(args.run)
    (args.reports/'stage2e_performance.json').write_bytes((json.dumps(summary,indent=2,sort_keys=True)+'\n').encode())
    (args.reports/'stage2e_performance.md').write_bytes(render(summary).encode())
    print(json.dumps({k:summary[k] for k in ('cache_parity_pass','cpu_gpu_parity_pass','go_full_cache_generation','go_production_training')}))
    print(json.dumps(summary['forecast'],indent=2))


if __name__=='__main__':
    main()
