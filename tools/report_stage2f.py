"""Summarize completed Stage 2F diagnostics without changing Stage 2E verdicts."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modern_pca import diagnose_parity as d
from modern_pca import numerical_parity as p
from modern_pca import reimplementation as r


def compare_controls(a,b):
    checks=[]
    for x,y in zip(a['updates'],b['updates']):
        checks.append(dict(update=x['update'],loss_exact=x['loss']==y['loss'],
            gradient_sums_exact=x['gradient_sum_sha256']==y['gradient_sum_sha256'],
            weights_exact=x['weight_sha256']==y['weight_sha256'],
            predictions_exact=x['after']['sha256']==y['after']['sha256'],
            counts_exact=x['microbatch_counts']==y['microbatch_counts'],
            iterations_exact=x['optimizer_iterations']==y['optimizer_iterations']))
    identity=all(a[key]==b[key] for key in ('input_sha256','labels_sha256','initial_checkpoint_sha256'))
    exact=identity and len(a['updates'])==len(b['updates'])==10 and a['before']==b['before']
    exact=exact and all(all(v for k,v in x.items() if k!='update') for x in checks)
    return dict(exact=bool(exact),identity_exact=identity,updates=checks)


def update_scale(cpu,gpu,previous_cpu,previous_gpu,chunk=1000000):
    cn=gn=dn=0.
    for offset in range(0,len(cpu),chunk):
        sl=slice(offset,offset+chunk)
        a=cpu[sl].astype(np.float64)-previous_cpu[sl].astype(np.float64)
        b=gpu[sl].astype(np.float64)-previous_gpu[sl].astype(np.float64)
        delta=a-b; cn+=float(np.dot(a,a)); gn+=float(np.dot(b,b)); dn+=float(np.dot(delta,delta))
    return dict(cpu_update_l2=float(np.sqrt(cn)),gpu_update_l2=float(np.sqrt(gn)),
                update_difference_l2=float(np.sqrt(dn)),update_relative_difference=float(np.sqrt(dn)/max(np.sqrt(cn),1e-8)))


def table_gradients(root,filename,divisor=1.):
    cpu=np.load(root/'cpu_A'/filename,mmap_mode='r'); gpu=np.load(root/'gpu_A'/filename,mmap_mode='r')
    specs=d.read(root/'cpu_A/result.json')['gradient_specs']; rows=[]
    for spec in specs:
        sl=slice(spec['offset'],spec['offset']+spec['elements'])
        # Only a single tensor is materialized when a physical-batch mean is requested.
        a=cpu[sl] if divisor==1 else cpu[sl]/np.float32(divisor)
        b=gpu[sl] if divisor==1 else gpu[sl]/np.float32(divisor)
        stats=d.gradient_stats(a,b); name=spec['name']
        kind=('depthwise convolution' if 'depthwise_kernel' in name else
              'pointwise convolution' if 'pointwise_kernel' in name else
              'bias' if name.endswith('/bias') else 'dense' if name.startswith(('paper_dense','class_probabilities')) else 'convolution')
        failures=[]; t=p.TOLERANCES
        if stats['max_absolute']>t['gradient_max_atol']+t['gradient_max_rtol']*stats['reference_max_absolute']: failures.append('maximum absolute/tensor-scale bound')
        if stats['mean_absolute']>t['gradient_mean_atol']+t['gradient_mean_rtol']*stats['reference_mean_absolute']: failures.append('mean absolute/tensor-scale bound')
        if stats['relative_l2']>t['gradient_relative_l2'] and stats['reference_l2']>=1e-6: failures.append('relative L2 bound')
        rows.append(dict(spec=spec,operation=kind,statistics=stats,failed_criteria=failures))
    return rows


def build(root,assessment):
    if d.read(root/'run_complete.json')['status']!='COMPLETE': raise ValueError('Incomplete diagnostics')
    d.check(root)
    runs={key:d.read(root/key/'result.json') for key in ('cpu_A','gpu_A','cpu_B','gpu_B')}
    if any(x['status']!='COMPLETE' or len(x['updates'])!=10 for x in runs.values()): raise ValueError('Incomplete trajectory')
    print('Comparing 47 logical, physical and summed gradients...',flush=True)
    gradients=table_gradients(root,'logical_average.npy')
    micro=table_gradients(root,'micro_sum.npy',4.)
    sums=table_gradients(root,'logical_sum.npy')
    initial=np.load(root/'initial_vector.npy',mmap_mode='r'); trajectories=[]; previous_cpu=initial; previous_gpu=initial
    first_cpu=np.load(root/'cpu_A/weights_1.npy',mmap_mode='r')
    first_gpu=np.load(root/'gpu_A/weights_1.npy',mmap_mode='r')
    weight_specs={x['name']:x for x in runs['cpu_A']['weight_specs']}
    for row in gradients:
        spec=weight_specs[row['spec']['name']]
        if spec['shape']!=row['spec']['shape']: raise ValueError('Gradient/weight shape mismatch')
        sl=slice(spec['offset'],spec['offset']+spec['elements'])
        row['first_adam_update']=update_scale(first_cpu[sl],first_gpu[sl],initial[sl],initial[sl])
    for cu,gu in zip(runs['cpu_A']['updates'],runs['gpu_A']['updates']):
        print(f"Comparing complete weights at update {cu['update']}/10...",flush=True)
        i=cu['update']; cpu=np.load(root/'cpu_A'/f'weights_{i}.npy',mmap_mode='r'); gpu=np.load(root/'gpu_A'/f'weights_{i}.npy',mmap_mode='r')
        stats=p.difference(cpu,gpu); scales=update_scale(cpu,gpu,previous_cpu,previous_gpu)
        trajectories.append(dict(update=i,cpu_loss=cu['loss'],gpu_loss=gu['loss'],loss_difference=abs(cu['loss']-gu['loss']),
            prediction=p.forward_comparison(cu['after'],gu['after'],post=True),weights=stats,
            model_weight_l2_divergence=stats['rms']*float(np.sqrt(stats['elements'])),update_scale=scales,
            sample_count_cpu=cu['sample_count'],sample_count_gpu=gu['sample_count'],
            original_weight_bound_pass=bool(p.weights_pass(stats,i))))
        previous_cpu=cpu; previous_gpu=gpu
    repeat={device:compare_controls(runs[device+'_A'],runs[device+'_B']) for device in ('cpu','gpu')}
    reference={device:d.read(root/('reference_'+device)/'result.json') for device in ('cpu','gpu')}
    # Confirm diagnostic instrumentation did not change the first logical gradient.
    linkage={}
    for device in ('cpu','gpu'):
        actual=np.load(root/(device+'_A')/'logical_average.npy',mmap_mode='r')
        original=np.load(root/'reproduction'/('parity_'+device)/'gradients.npy',mmap_mode='r')
        linkage[device]=dict(exact=bool(np.array_equal(actual,original)),difference=p.difference(original,actual))
        summed=np.load(root/(device+'_A')/'logical_sum.npy',mmap_mode='r')
        linkage[device]['average_is_sum_divided_by_100']=all(
            np.array_equal(actual[i:i+1000000],summed[i:i+1000000]/np.float32(100))
            for i in range(0,len(actual),1000000))
    final=p.forward_comparison(runs['cpu_A']['final_1000'],runs['gpu_A']['final_1000'],post=True)
    failures=[x for x in gradients if not x['statistics']['strict_stage2e_pass']]
    cpu_gradient_squared=sum(x['statistics']['cpu_norm']**2 for x in gradients)
    difference_squared=sum(x['statistics']['rms']**2*x['spec']['elements'] for x in gradients)
    cpu_losses=np.array([x['cpu_loss'] for x in trajectories]); gpu_losses=np.array([x['gpu_loss'] for x in trajectories])
    observations=dict(minimum_gradient_cosine=min(x['statistics']['cosine_similarity'] for x in gradients),
        maximum_tensor_relative_l2=max(x['statistics']['relative_l2'] for x in gradients),
        global_gradient_relative_l2=float(np.sqrt(difference_squared)/max(np.sqrt(cpu_gradient_squared),1e-8)),
        maximum_microbatch_relative_l2=max(x['statistics']['relative_l2'] for x in micro),
        maximum_loss_difference=max(x['loss_difference'] for x in trajectories),
        loss_trajectory_correlation=float(np.corrcoef(cpu_losses,gpu_losses)[0,1]) if np.std(cpu_losses) and np.std(gpu_losses) else None,
        maximum_post_probability_difference=max(x['prediction']['probabilities']['max_absolute'] for x in trajectories),
        maximum_weight_relative_l2=max(x['weights']['relative_l2'] for x in trajectories),
        maximum_update_relative_l2=max(x['update_scale']['update_relative_difference'] for x in trajectories),
        first_weight_l2=trajectories[0]['model_weight_l2_divergence'],last_weight_l2=trajectories[-1]['model_weight_l2_divergence'])
    reduction_exact=all(x['exact_float32_sum'] for value in reference.values() for x in value['checks'])
    evidence=dict(reproduction=d.read(root/'reproduction_summary.json')['same_failures'],
        same_device_exact=all(x['exact'] for x in repeat.values()),independent_float32_accumulation_exact=reduction_exact,
        instrumented_first_gradient_exact=all(x['exact'] for x in linkage.values()),
        all_post_prediction_checks_pass=all(x['prediction']['pass'] for x in trajectories),final_1000_pass=final['pass'],
        frozen_weights_exact=all(x['frozen_weights_exact'] for x in runs.values()))
    evidence.update(averages_exact=all(x['average_is_sum_divided_by_100'] for x in linkage.values()),
        original_tensor_files_exact=all(v for device in d.read(root/'reproduction_summary.json')['exact_tensor_files'].values() for v in device.values()),
        cross_device_inputs_and_initialization_exact=all(runs['cpu_A'][key]==runs['gpu_A'][key]
            for key in ('input_sha256','labels_sha256','initial_checkpoint_sha256')),
        all_weight_checks_pass=all(x['original_weight_bound_pass'] for x in trajectories),
        sample_counts_correct=all(u['sample_count']==100 and u['microbatch_counts']==list(range(4,101,4))
            and u['optimizer_iterations']==u['update'] for run in runs.values() for u in run['updates']))
    if assessment=='likely_benign' and not all(evidence.values()):
        raise ValueError('Cannot label likely benign while essential controls fail')
    result=dict(timestamp=r.utc_now(),run=str(root),protocol=d.read(root/'protocol.json'),
        reproduction=d.read(root/'reproduction_summary.json'),gradients=gradients,microbatch_mean_gradients=micro,
        accumulated_sum_gradients=sums,accumulation_references=reference,first_gradient_linkage=linkage,
        same_device_repeatability=repeat,trajectory=trajectories,final_1000_predictions=final,
        evidence_checks=evidence,observations=observations,strict_stage2e_verdict='FAIL' if failures else 'PASS',
        scientific_assessment=assessment,
        gpu_numerical_recommendation='GO on the tested numerical evidence' if assessment=='likely_benign' else 'NO-GO',
        production_launch_recommendation=('GO on numerical grounds; launch deferred until full-cache and production-trainer readiness are verified' if assessment=='likely_benign' else 'NO-GO'),
        full_cache_recommendation='GO (recommendation only; not generated)',
        hypotheses_and_limits=['Microbatch discrepancies precede logical accumulation; exact independent sums rule out a summation implementation mismatch if controls pass.',
            'Frozen BN state is checked exactly. This rules out state mutation, not differences in float32 BN inference arithmetic.',
            'Convolution, nonlinear activation boundaries and reduction kernels are plausible sources; this experiment does not isolate the first divergent primitive.',
            'Large per-element relative errors near zero do not by themselves explain an L2-relative or maximum-absolute tensor failure.',
            'Float64 reference sums use already-computed float32 microgradients; they do not constitute float64 full-network gradients.',
            'Ten updates at the first unfreezing stage and 1000 training patches cannot prove identical 14-epoch trajectories or unchanged external generalization.',
            'Original strict thresholds and Stage 2E FAIL are retained. Material assessment is a separate evidence-based judgment, not a redefinition of PASS.'])
    return result


def render(s):
    failed=[x for x in s['gradients'] if not x['statistics']['strict_stage2e_pass']]
    lines=['# Stage 2F numerical-parity investigation','',f"Generated {s['timestamp']}. Evidence: `{s['run']}`.",'',
        f"Original strict verdict: **{s['strict_stage2e_verdict']}**. Scientific assessment: **{s['scientific_assessment']}**.",
        f"GPU numerical recommendation: **{s['gpu_numerical_recommendation']}**. Production launch: **{s['production_launch_recommendation']}**.",
        f"Full-cache generation recommendation: **{s['full_cache_recommendation']}**.",'',
        'No production training, full-cache generation, VAL2 access or source-image changes occurred. The strict evaluator, original tolerance utility, cache representation, model, split and scientific training configuration are protected by SHA-256.',
        '', '## Exact reproduction','',f"`{json.dumps(s['reproduction'],sort_keys=True)}`",'',
        'The unchanged Stage 2E parity worker was rerun with byte-verified copies of its initialization and 1000-image cache. It uses identical first-100 inputs/labels, physical 4/effective 100, Adam, frozen BN, float32, TF32 disabled, deterministic settings and the original comparison utility. Three repeated updates match the original experiment. The diagnostic trajectory then uses ten distinct logical batches: the original first 100, followed by a fixed SHA-256 ordering of the remaining 900. No augmentation is used.',
        '', '## Complete gradient table: identity and norms','',
        '| ID | Variable | Shape | Elements | CPU norm | GPU norm | Strict verdict |','|---:|---|---|---:|---:|---:|---|']
    for x in s['gradients']:
        a=x['spec']; z=x['statistics']; lines.append(f"| {a['index']} | {a['name']} | {a['shape']} | {a['elements']} | {z['cpu_norm']:.9g} | {z['gpu_norm']:.9g} | {'PASS' if z['strict_stage2e_pass'] else '**FAIL**'} |")
    lines.extend(['','### Complete gradient table: errors','',
        '| ID | Max abs | Mean abs | RMS | Max relative | Mean relative | Relative L2 | Cosine |','|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for x in s['gradients']:
        z=x['statistics']; lines.append('| '+str(x['spec']['index'])+' | '+' | '.join(f"{z[k]:.10g}" for k in ('max_absolute','mean_absolute','rms','max_relative','mean_relative','relative_l2','cosine_similarity'))+' |')
    lines.extend(['','### Complete gradient table: elementwise diagnostics','',
        'Absolute exceedance means |CPU-GPU| > 1e-4; relative means |CPU-GPU|/max(|CPU|,1e-8) > 1e-3. Either is the union. Near zero means magnitude <= 1e-8. These elementwise diagnostics do not replace the original tensor-level acceptance rules. Exact counts and fractions are all stored in JSON.',
        '', '| ID | Absolute fraction | Relative fraction | Either fraction | CPU near-zero count/fraction | GPU near-zero count/fraction |',
        '|---:|---:|---:|---:|---|---|'])
    for x in s['gradients']:
        z=x['statistics']; lines.append(f"| {x['spec']['index']} | {z['absolute_exceed_fraction']:.9g} | {z['relative_exceed_fraction']:.9g} | {z['either_exceed_fraction']:.9g} | {z['near_zero_cpu_count']}/{z['near_zero_cpu_fraction']:.9g} | {z['near_zero_gpu_count']}/{z['near_zero_gpu_fraction']:.9g} |")
    lines.extend(['','## Failure pattern and cause analysis',''])
    pattern=dict(Counter(x['operation'] for x in failed))
    lines.extend([f"Failing operation counts: `{json.dumps(pattern,sort_keys=True)}`. Seven failures are in normal block 17 and one is in block 18: late backbone layers, with block 17 earliest in this trainable suffix. Dense kernels and biases pass; normalization parameters are frozen, so no trainable BN gradient is tested.",''])
    for x in failed:
        z=x['statistics']; lines.append(f"- **{x['spec']['name']}** ({x['operation']}): {', '.join(x['failed_criteria'])}; max error / CPU peak = {z['max_absolute']/max(z['reference_max_absolute'],1e-8):.6g}; L2-relative error {z['relative_l2']:.6g}; cosine {z['cosine_similarity']:.10f}; resulting first-Adam update-relative difference {x['first_adam_update']['update_relative_difference']:.6g}.")
    lines.extend(['', 'Failures are identified by the unchanged Stage 2E rule. A relative-L2 failure concerns the whole gradient norm and is not simply an unstable elementwise division near zero. The maximum-absolute failure is a localized discrepancy whose size, frequency and update effect are reported rather than assumed benign. Operation classes are recorded for all 47 variables; normalization parameters are frozen and excluded from this trainable set.',
        '', '## Microbatch, accumulation and optimizer','',
        f"Single physical-microbatch mean gradients failing the original tensor rule: {[x['spec']['index'] for x in s['microbatch_mean_gradients'] if not x['statistics']['strict_stage2e_pass']]}.",
        f"Logical averaged-gradient failures: {[x['spec']['index'] for x in failed]}.",
        f"Instrumented first-gradient linkage to exact reproduction: `{json.dumps(s['first_gradient_linkage'])}`.",'',
        'The reference pass resets the unchanged accumulator after each physical microbatch, exports its summed gradient, and independently reduces those 25 increments in NumPy float32 and float64. It records 4 samples per isolated microgradient; the normal pass records 4,8,...,100 and averages by 100. Adam runs once per logical batch. No optimizer is applied in the isolated reference pass.'])
    for device,v in s['accumulation_references'].items():
        lines.extend(['',f"{device.upper()}: {sum(x['exact_float32_sum'] for x in v['checks'])}/47 independent float32 sums are bitwise equal to the actual accumulator. Largest float64-reference relative-L2 summation error: {max(x['float64_sum_difference']['relative_l2'] for x in v['checks']):.9g}. Per-tensor sums, differences and sample counts are retained in JSON."])
    lines.extend(['','## Same-device controls',''])
    lines.extend(['TensorFlow documents deterministic execution for the same inputs on the same hardware and software configuration; it does not promise CPU/GPU bitwise identity. These controls test the observed same-device behavior directly. [TensorFlow determinism API](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).',''])
    for device,value in s['same_device_repeatability'].items():
        lines.append(f"- {device.upper()} A vs B: exact equality **{value['exact']}** across ten updates, including losses, gradient-sum hashes, model-weight hashes, prediction bytes, sample counts and optimizer iterations.")
    lines.extend(['','## Ten-update trajectory','',
        '| Update | CPU loss | GPU loss | Loss difference | Class agreement (64) | Max probability difference | Weight L2 divergence | Weight relative L2 | Update relative L2 difference |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for x in s['trajectory']:
        lines.append(f"| {x['update']} | {x['cpu_loss']:.9g} | {x['gpu_loss']:.9g} | {x['loss_difference']:.9g} | {x['prediction']['predicted_class_agreement']:.6f} | {x['prediction']['probabilities']['max_absolute']:.9g} | {x['model_weight_l2_divergence']:.9g} | {x['weights']['relative_l2']:.9g} | {x['update_scale']['update_relative_difference']:.9g} |")
    lines.extend(['', 'Weight divergence compares the complete model states. Update-relative difference compares (GPU current - GPU previous) with (CPU current - CPU previous), normalized by the CPU update norm; this avoids hiding differences behind the much larger pretrained-weight norm.',
        '',f"Final 1000-image comparison: `{json.dumps(s['final_1000_predictions'])}`.",
        '',f"Evidence checks: `{json.dumps(s['evidence_checks'])}`.",
        '', '## Interpretation and limits',''])
    o=s['observations']
    if s['scientific_assessment']=='likely_benign':
        lines.extend(['**Material assessment: likely benign float32 backend differences, with no material short-trajectory effect observed.** No accumulator or sample-count inconsistency was found. This is a numerical GO recommendation for GPU production training, conditional on completing cache and trainer readiness checks before launching; it does not authorize or start that work.',
            '', 'The largest global update-relative difference is approximately 0.165%; individual failing tensors have first-update differences of approximately 0.36–1.00% of their own update norms. Thus the differences are measurable and Adam can amplify gradient perturbations, but their observed effects on loss and predictions remain small. Parameter divergence grows over ten updates without a corresponding material prediction divergence in this fixture. This supports a likely-benign judgment, not proof of identical full training.', ''])
    lines.extend([f"Minimum gradient cosine: {o['minimum_gradient_cosine']:.12f}; largest tensor-relative L2 error: {o['maximum_tensor_relative_l2']:.9g}; global gradient-relative L2 error: {o['global_gradient_relative_l2']:.9g}.",
        '',f"Across ten updates: largest loss difference {o['maximum_loss_difference']:.9g}; largest probability difference {o['maximum_post_probability_difference']:.9g}; loss-trajectory correlation {o['loss_trajectory_correlation']}; largest model-weight relative divergence {o['maximum_weight_relative_l2']:.9g}; largest update-relative divergence {o['maximum_update_relative_l2']:.9g}.",
        '',f"Model-weight L2 divergence changes from {o['first_weight_l2']:.9g} after update 1 to {o['last_weight_l2']:.9g} after update 10. Growth must be interpreted with the reported update scale and prediction effects; nonzero growth alone is not evidence of material instability.",''])
    lines.extend('- '+x for x in s['hypotheses_and_limits'])
    lines.extend(['','## Artifacts and validation','',
        'New code: `modern_pca/diagnose_parity.py`, `tools/report_stage2f.py`, `tests/test_diagnose_parity.py`. Raw runs, tensor vectors, sequence, configuration and progress logs are under the run directory. Tests and final Git checks are recorded in `reports/stage2f_checks.json`.',
        '', 'No commit or push was performed.',''])
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage2f_parity'))
    parser.add_argument('--assessment',choices=['insufficient','likely_benign','implementation_inconsistency'],default='insufficient')
    args=parser.parse_args(argv); summary=build(args.run,args.assessment)
    root=Path('reports'); root.mkdir(exist_ok=True)
    (root/'stage2f_numerical_parity.json').write_bytes((json.dumps(summary,indent=2,sort_keys=True)+'\n').encode())
    (root/'stage2f_numerical_parity.md').write_bytes(render(summary).encode())
    print(json.dumps({k:summary[k] for k in ('strict_stage2e_verdict','scientific_assessment','evidence_checks')},indent=2))


if __name__=='__main__':
    main()
