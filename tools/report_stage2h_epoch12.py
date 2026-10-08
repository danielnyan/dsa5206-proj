"""Assemble bounded Stage 2H OOM evidence; never executes a resume command."""
import json
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


def main():
    root=Path('runs/stage2h_epoch12_oom_smoke_20261007')
    read=lambda p: json.loads(p.read_text())
    results={p.parent.name:read(p) for p in root.glob('*/result.json')}
    results.pop('baseline_transitions_12_14', None)  # Excluded instrumentation-retention trial.
    if 'baseline_transitions_12_14_clean' in results:
        results['baseline_transitions_12_14']=results['baseline_transitions_12_14_clean']
    required=[f'{c}_epoch{e}' for c in ('A','B','C') for e in (12,13,14)]
    required+=['baseline_epoch12']+[f'{c}_fixture' for c in ('baseline','A','B','C')]
    required+=['baseline_transitions_12_14','A_transitions_12_14']
    assert all(k in results for k in required), 'Investigation still running'
    arrays={k:np.load(root/f'{k}_fixture/fixture_outputs.npz') for k in ('baseline','A','B','C')}
    equivalence={}
    for candidate in ('A','B','C'):
        diffs={key:float(np.max(np.abs(arrays['baseline'][key]-arrays[candidate][key])))
               for key in arrays['baseline'].files}
        for key in arrays['baseline'].files:
            np.testing.assert_allclose(arrays[candidate][key],arrays['baseline'][key],
                                       atol=1e-6 if key.startswith('g') else 1e-7,rtol=1e-5)
        equivalence[candidate]=dict(status='PASS', max_absolute_gradient_difference=max(v for k,v in diffs.items() if k.startswith('g')),
            max_absolute_weight_difference=max(v for k,v in diffs.items() if k.startswith('w')),
            max_absolute_prediction_difference=diffs['predictions'],
            class_predictions_identical=bool(np.array_equal(arrays['baseline']['predictions'].argmax(1),
                                                            arrays[candidate]['predictions'].argmax(1))))
    baseline=results['baseline_epoch12']; async12=results['A_epoch12']
    real_equivalence=dict(same_sample_ids=baseline['sample_ids']==async12['sample_ids'],
        identical_weight_hashes=baseline.get('model_weight_hashes_after')==async12.get('model_weight_hashes_after'),
        max_absolute_validation_probability_difference=float(np.max(np.abs(np.asarray(baseline['validation_scores'])-
                                                                          np.asarray(async12['validation_scores'])))),
        max_absolute_loss_difference=max(abs(a['loss']-b['loss']) for a,b in zip(baseline['logical_updates'],async12['logical_updates'])))
    full_model_other={}
    for candidate in ('B','C'):
        value=results[f'{candidate}_epoch12']
        if value['status']=='PASS':
            full_model_other[candidate]=dict(identical_weight_hashes=baseline['model_weight_hashes_after']==value['model_weight_hashes_after'],
                max_absolute_validation_probability_difference=float(np.max(np.abs(np.asarray(baseline['validation_scores'])-
                                                                                  np.asarray(value['validation_scores'])))),
                max_absolute_loss_difference=max(abs(a['loss']-b['loss']) for a,b in zip(baseline['logical_updates'],value['logical_updates'])))
    for epoch in (12,13,14):
        reference=results[f'A_epoch{epoch}']
        for candidate in ('B','C'):
            value=results[f'{candidate}_epoch{epoch}']
            assert reference['variables']==value['variables']
            assert reference['sample_ids']==value['sample_ids']
            assert reference['fresh_adam_initial_iterations']==value['fresh_adam_initial_iterations']==0
    recommended_pass=all(results[k]['status']=='PASS' for k in ('A_epoch12','A_epoch13','A_epoch14','A_fixture','A_transitions_12_14'))
    assert recommended_pass and equivalence['A']['class_predictions_identical']
    assert all(v['original_artifacts_unchanged'] and v['VAL2_access_attempts']==0 for v in results.values())
    transition_equivalence={}
    for epoch in (12,13,14):
        if (root/'baseline_transitions_12_14_clean'/f'epoch{epoch}_disposable/checkpoint.json').exists():
            original=read(root/'baseline_transitions_12_14_clean'/f'epoch{epoch}_disposable/checkpoint.json')
            async_value=read(root/'A_transitions_12_14'/f'epoch{epoch}_disposable/checkpoint.json')
            transition_equivalence[str(epoch)]=dict(
                model_hashes_identical=original['model_tensor_sha256']==async_value['model_tensor_sha256'],
                optimizer_hashes_identical=original['optimizer_tensor_sha256']==async_value['optimizer_tensor_sha256'])
    fresh=Path('runs/stage2h_production_resume_epoch11_cuda_malloc_async')
    assert not fresh.exists()
    command="""cd '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj'
TF_GPU_ALLOCATOR=cuda_malloc_async \\
  /home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python -B -u \\
  -m modern_pca.train_reimplementation \\
  --train-production \\
  --manifest manifests/VAL1_manifest.csv \\
  --split-output manifests/VAL1_internal_v1 \\
  --cache runs/stage2g_full_cache/cache \\
  --batch-size 4 \\
  --validation-batch-size 2 \\
  --resume-cache-checkpoint runs/stage2h_production_resume_epoch08_vbatch2/epoch_11_resume \\
  --output runs/stage2h_production_resume_epoch11_cuda_malloc_async"""
    table=[]
    for candidate in ('A','B','C'):
        rows=[results[f'{candidate}_epoch{e}'] for e in (12,13,14)]
        peaks=[v.get('memory_after_validation',v.get('memory_at_failure',{})).get('peak',0) for v in rows]
        speeds=[v.get('steady_images_per_second') for v in rows]
        table.append(dict(candidate=candidate,epochs={str(e):results[f'{candidate}_epoch{e}']['status'] for e in (12,13,14)},
                          peak_bytes=max(peaks),steady_images_per_second=speeds,numerical=equivalence[candidate],
                          protocol_deviation={'A':'Allocator only; training 4/effective100, validation2 unchanged',
                                              'B':'Physical training batch2 instead of frozen4; effective100 unchanged',
                                              'C':'Experimental serialized buffer averaging; Adam mathematics unchanged'}[candidate]))
    report=dict(status='PASS',recommendation='A: batch4 + cuda_malloc_async',resume_command=command,
                environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                    packages={name:importlib.metadata.version(name) for name in
                        ('tensorflow','keras','numpy','Pillow','staintools','opencv-python','spams-bin')}),
                production_training_started=False,VAL2_images_accessed=False,commit_push_performed=False,
                VAL2_audit=dict(probe_and_test_real_access_attempts=0,images_opened_or_evaluated=False,
                    git_manifest_inspection=True,
                    explanation='Repository-wide git diff --check inspected tracked VAL2 manifest files; this was outside the requested no-VAL2 scope. No VAL2 images or inference were used, and no manifest contents informed recovery decisions.'),
                original_artifacts_unchanged=True,comparison=table,fixture_equivalence=equivalence,
                full_model_baseline_async_equivalence=real_equivalence,full_model_other_candidates=full_model_other,
                transition_checkpoint_equivalence=transition_equivalence,
                baseline_transition_failure={key: results['baseline_transitions_12_14'].get(key)
                    for key in ('status', 'phase', 'memory_at_failure', 'error')},
                same_trainable_variables_and_sample_order=True,results=results,
                original_failure=dict(boundary='normal_conv_1_5',trainable_parameters=206213042,
                    checkpoint_completed_utc='2026-10-06T17:37:31.036185+00:00',failure_log_time='2026-10-07 01:38:18.192',
                    op='truediv_331',variable='paper_dense_256/kernel',shape=[487872,256],bytes=499580928,
                    epoch_start_precision='After completed epoch11 checkpoint and before failure; no explicit start timestamp in original log',
                    sample_cursor='100..1000 samples (updates1..10); progress logs occur after a successful flush. First flush likely, not provable from existing log.',
                    original_allocator_stats_source='User-provided console summary: ~9.968GB limit, ~9.35GB InUse, ~9.95GB MaxInUse, LargestFreeBlock=0'),
                scope='Fresh-process stages and bounded same-process transitions; not a full-epoch or long-duration OOM guarantee')
    test_results={}
    for name in ('focused', 'full'):
        path=Path(f'reports/stage2h_epoch12_{name}_tests.log')
        summaries=[line for line in path.read_text().splitlines() if ' passed' in line]
        assert summaries and 'failed' not in summaries[-1], f'{name} tests not complete'
        test_results[name]=dict(log=str(path), summary=summaries[-1])
    tracked=['modern_pca/cached_training.py','modern_pca/train_reimplementation.py','tests/test_stage2h_readiness.py']
    added=['tools/stage2h_epoch12_probe.py','tools/report_stage2h_epoch12.py','tests/test_stage2h_oom_candidates.py']
    checks=[subprocess.run(['git','diff','--check','--',*tracked],capture_output=True,text=True)]
    checks += [subprocess.run(['git','diff','--no-index','--check','/dev/null',name],capture_output=True,text=True) for name in added]
    # --no-index implies --exit-code: a new file differs from /dev/null (1).
    # Whitespace errors are emitted to stdout; usage/execution errors are >1.
    assert checks[0].returncode==0 and all(check.returncode in (0,1) and not check.stdout
                                         for check in checks[1:]), 'Whitespace check failed'
    report['repository_checks']=dict(tests=test_results,diff_check='PASS',
        pytest_exit_warning='After both suites passed with exit0 and zero recorded real VAL2 attempts, pytest atexit temporary-directory cleanup was blocked by the conservative audit guard on relative VAL2-named entries. No blocked open proceeded; the warning is separate from test results.',
        diff_check_output=''.join(check.stdout+check.stderr for check in checks),
        git_status=Path('reports/stage2h_epoch12_git_status.txt').read_text(encoding='utf-8-sig'),
        git_status_source='Windows Git in the native working tree; WSL Git has different line-ending configuration',
        tracked_diff_stat=subprocess.check_output(['git','diff','--stat','--',*tracked],text=True),
        files_added_this_investigation=added,
        files_modified_this_investigation=['tests/test_stage2h_readiness.py'],
        preexisting_tracked_changes=tracked)
    destination=Path('reports/stage2h_epoch12_oom_investigation.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Stage 2H epoch-12 OOM investigation','',
        '**Recommendation: A — physical batch 4 + `TF_GPU_ALLOCATOR=cuda_malloc_async`; effective batch 100 and validation batch 2.**','',
        'No production run was started. All optimizer updates belong to bounded disposable smoke tests. Existing production checkpoints/history and cache metadata stayed unchanged. GPU probes and tests recorded zero real VAL2 access attempts; no VAL2 images were opened or evaluated. A repository-wide git diff --check did inspect tracked VAL2 manifest files, contrary to the requested no-VAL2 scope. Their contents were not used in recovery decisions. No commit/push was performed.','',
        '## Root cause and recovery state','',
        'Epoch 11 is complete at `normal_conv_1_7`, Adam iteration 1167. Strict restoration verifies checkpoint-file, model-tensor and optimizer-tensor hashes. The next production epoch is 12, boundary `normal_conv_1_5`, LR 1e-6, with **206,213,042 trainable parameters** and a fresh Adam optimizer.','',
        'The failing operation `truediv_331` is the accumulator buffer for `paper_dense_256/kernel` divided by the actual logical sample count. Its `[487872,256]` float32 result requires **499,580,928 bytes = 476.44 MiB**. The existing list-based apply materializes averaged gradient tensors while model weights, accumulated sums and Adam slots remain resident. Adam slots are built lazily at the first apply.','',
        'The original run has no explicit epoch-start or per-flush cursor. Epoch 12 starts after checkpoint completion at 2026-10-07 01:37:31.036 Singapore time and fails at 01:38:18.192. No 1000-sample progress message appears: the failing flush corresponds to 100–1000 samples (updates1–10), because progress is logged only after a successful flush. The first flush is plausible but cannot be established exactly from these logs.','',
        'The user-provided allocator summary reports ~9.35GB in use against a ~9.968GB pool, a ~9.95GB high-water mark and no largest free block. The failed request is ~0.500GB. This supports fragmentation/allocator-history pressure combined with high live memory, rather than proving that the frozen epoch-12 model inherently exceeds capacity. The fresh-process baseline completed its bounded updates, so the original OOM did **not** reproduce there. No claim is made to have replicated hours of allocator history.','',
        'At normal transitions, the source deletes the accumulator/optimizer and calls `gc.collect()`. Cross-stage resume deletes the old Adam after strict restoration. Weak-reference observations and GPU memory snapshots are recorded per test; no speculative production lifetime/allocator code was changed.','',
        '## Candidate results','',
        '| Candidate | Epoch12 | Epoch13 | Epoch14 | Max GPU peak (GB) | Steady images/s (12 / 13 / 14) | Numerical equivalence | Protocol deviation | Recommendation |',
        '|---|---|---|---|---:|---|---|---|---|']
    for row in table:
        e=row['epochs']; speed=' / '.join('—' if x is None else f'{x:.2f}' for x in row['steady_images_per_second'])
        lines.append(f"| {row['candidate']} | {e['12']} | {e['13']} | {e['14']} | {row['peak_bytes']/1e9:.3f} | {speed} | PASS, quantified below | {row['protocol_deviation']} | {'Recommended' if row['candidate']=='A' else 'Not selected'} |")
    lines+=['','Every independent stage smoke restores epoch 11 and uses 300 deterministic epoch-specific VAL1 training samples: 75 microbatches at physical4, or 150 at physical2, and exactly three 100-sample updates. Epoch13/14 independent probes use epoch11 weights as disposable initial test state, not claims of completed production epochs. Each successful stage also checks inference on eight internal-validation samples at batch2.','',
            'Same-process transition checks additionally carry disposable updated weights through stages12→13→14 with fresh Adam at each stage, without resetting the allocator or clearing the session. Each stage uses **355 samples (100,100,100,55)**, exercising the final short microbatch/retrace, inference and full disposable checkpoint writing before the next stage. The final disposable model/Adam checkpoint is strictly restored again:']
    for name in ('baseline_transitions_12_14','A_transitions_12_14'):
        value=results[name]
        lines.append(f"- `{name}`: **{value['status']}**; transition details and memory snapshots in JSON.")
        stages=[dict(epoch=12, memory_after_training=value['memory_after_training'])]+value.get('transition_stages',[])
        for stage in stages:
            lines.append(f"  - Stage {stage['epoch']}: training memory current/peak bytes `{stage.get('memory_after_training')}`; object lifetimes `{stage.get('lifetime', 'not instrumented')}`.")
    failure=results['baseline_transitions_12_14']
    if failure['status']=='OOM':
        lines+=['',f"Corrected baseline transition failure: phase `{failure['phase']}`, current/peak bytes `{failure['memory_at_failure']}`. Stage 14 completed its first optimizer update, then failed in the next microbatch at `AssignAddVariableOp_419`, accumulating the `[487872,256]` dense tensor. This reproduces transition-related memory pressure, but is a different operation/stage from the original epoch-12 division OOM. It does not establish fragmentation as the sole cause."]
    lines+=['','In the async transition probe, old accumulator and optimizer weak references were both dead after collection, yet live GPU allocation decreased by only 304 bytes at each transition. Python object collection therefore does not imply release of all TensorFlow runtime resources. The remaining ownership was not isolated; attributing the growth solely to fragmentation or a specific leaked Python optimizer would exceed the evidence. No production resource-lifetime code was changed.']
    lines+=['','## Numerical checks','',
        'The small deterministic frozen-layer/BN fixture uses 355 samples: effective groups **100,100,100,55**, four updates, no dropped samples, finite gradients/losses, frozen-weight hashes and checkpoint save/perturb/restore verification. Predeclared tolerances: gradients atol1e-6/rtol1e-5; weights/predictions atol1e-7/rtol1e-5. Tolerances were not loosened.','',
        '| Candidate vs baseline fixture | Max absolute gradient difference | Max absolute weight difference | Max absolute prediction difference | Class agreement |',
        '|---|---:|---:|---:|---|']
    for k,v in equivalence.items():
        lines.append(f"| {k} | {v['max_absolute_gradient_difference']:.9g} | {v['max_absolute_weight_difference']:.9g} | {v['max_absolute_prediction_difference']:.9g} | {v['class_predictions_identical']} |")
    lines+=['',f'Full NASNet epoch12 baseline vs async: `{json.dumps(real_equivalence)}`.',
        f'Same-process transition checkpoint model/Adam hash comparisons: `{json.dumps(transition_equivalence)}`.',
        f'Other epoch12 full-model comparisons: `{json.dumps(full_model_other)}`. Full weight hashes measure bitwise agreement; magnitude of B weight differences is measured on the deterministic fixture, not inferred from hashes.',
        'Exact trainable variable names/shapes, ordered sample IDs and zero initial new-stage Adam iterations are checked across A/B/C at every stage.','',
        '## Tests and working tree','',
        f"Focused: {test_results['focused']['summary']}. Full: {test_results['full']['summary']}. Both run CPU-only after GPU probes, with real VAL2 paths blocked (synthetic pytest fixtures allowed).",
        'Git diff --check: PASS, including no-index whitespace checks for the new Python files. Existing tracked validation-batch changes were preserved. This investigation adds diagnostic tools, tests and reports, and extends the existing resume orchestration test to cover epoch11 -> epoch12 with history1..12 exactly once. No production source was changed in this investigation.','',
        '```text',report['repository_checks']['git_status'],'```','',
        '## Warnings and limitations','',
        'After the focused/full tests completed successfully, pytest atexit temporary-directory cleanup emitted two audit-guard exceptions on relative VAL2-named entries. The conservative guard blocked those opens; no blocked open proceeded. The suites had already reported 64 and 254 passing tests, zero real VAL2 attempts and exit0. This cleanup warning is retained separately from the test outcomes.','',
        'Console logs retain TensorFlow startup/oneDNN notices and any cuDNN heuristic fallback warnings. A fallback warning occurred in the batch2 epoch13 run; the test still completed with finite losses/gradients. No warnings were suppressed and no numerical tolerances were relaxed. Candidate C is a tested hypothesis, not an assumed improvement: its serialized averaging may have a higher actual peak/compile cost despite reducing an explicit simultaneous gradient list.','',
        'An initial baseline transition diagnostic retained an inspection-only ConcreteFunction reference across stages. That artificial resource retention was identified, the probe interrupted and preserved, and the reference removed. Only the corrected `baseline_transitions_12_14_clean` results are used here; the async transition probe also uses the corrected harness.','',
        '## Reproduction commands and environment','',
        'Environment/build/GPU details, allocator environment, exact worker argument lists, sample IDs, checkpoint/history hashes, loss per update, compile-inclusive and steady timings, current/peak memory and error tracebacks are in the JSON report. Console logs retain allocator startup messages and warnings.','',
        'From the repository in WSL, each worker was invoked with the verified environment Python and:','',
        '```text','python -B -u -m tools.stage2h_epoch12_probe --candidate {baseline,A,B,C} --epoch {12,13,14} --updates 3 --output <new isolated directory>',
        '# A only: TF_GPU_ALLOCATOR=cuda_malloc_async',
        '# Numerical fixtures: --fixture instead of an epoch test',
        '# Same-process transition tests: --epoch 12 --chain', '```','',
        'Outputs are under `runs/stage2h_epoch12_oom_smoke_20261007/`. Candidate C exists only in the diagnostic harness and is not wired into production. Candidate B bypasses the production CLI only within the bounded diagnostic.','',
        '## Proposed production resume — NOT EXECUTED','',
        'The command resumes epoch12 from the latest complete epoch11 checkpoint. It uses a new output directory. No architecture, preprocessing, sample order, augmentation, Adam mathematics, LR, schedule, BN policy or batch semantics change.','',
        '```bash',command,'```','',
        'Bounded feasibility is not a guarantee against an OOM after an entire long-running epoch. The recommendation is supported by all remaining stages and the same-process transition smoke, with no physical-batch protocol deviation.','']
    Path('reports/stage2h_epoch12_oom_investigation.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=report['status'],comparison=table,full_model_equivalence=real_equivalence),indent=2))


if __name__=='__main__': main()
