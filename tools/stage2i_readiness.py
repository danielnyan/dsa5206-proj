"""Read-only Stage 2I verification. No final evaluation or training entry point."""
import argparse
from collections import Counter
import contextlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
from modern_pca import reimplementation as r, cached_training as c
from modern_pca import evaluate_reimplementation as e, stage2i_contract as f

RUN = Path('runs/stage2h_production_resume_epoch11_cuda_malloc_async')
CHECKPOINT = RUN/'epoch_14_resume'
EXTRACTED = Path('/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/zenodo/_3825933/extracted')
REPORTS = Path('reports')


def guard():
    blocked=[]
    def audit(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=os.fsdecode(args[0]).replace('\\','/').lower()
        # Synthetic pytest fixtures are permitted; actual VAL2 roots never are.
        if 'val_dataset_2' in path and not path.startswith('/tmp/stage2i_tests_'):
            blocked.append(path)
            raise RuntimeError('VAL2 image/root access forbidden during readiness')
    sys.addaudithook(audit)
    return blocked


def read(path):
    return json.loads(Path(path).read_text())


def snapshot():
    paths=[*CHECKPOINT.iterdir(),RUN/'history.json',RUN/'metadata.json',RUN/'epoch14.keras']
    return {str(path):r.sha256_file(path) for path in paths if path.is_file()}


def disable_training(tf):
    def forbidden(*args,**kwargs):
        raise AssertionError('Training/optimizer update forbidden in Stage 2I readiness')
    tf.keras.optimizers.Adam.apply_gradients=forbidden
    r.Accumulator.add=forbidden
    r.Accumulator.flush=forbidden
    c.train_batches=forbidden


def restore():
    import tensorflow as tf
    disable_training(tf)
    device=r.gpu_setup()
    logging.info('Hashing final checkpoint/export/history before strict restoration')
    before=snapshot()
    saved=read(CHECKPOINT/'checkpoint.json')
    assert set(p.name for p in CHECKPOINT.iterdir()) == {'checkpoint.json','state.index','state.data-00000-of-00001'}
    assert set(saved['files_sha256']) == {'state.index','state.data-00000-of-00001'}
    c.validate_finalization_state(saved['state'])
    history=read(RUN/'history.json')
    assert history['selection']=='none; fixed epoch14'
    assert history['epochs']==saved['state']['history']
    assert [v['epoch'] for v in history['epochs']]==list(range(1,15))
    sidecar=read(RUN/'metadata.json')
    e.validate_sidecar(sidecar,RUN/'epoch14.keras',True)
    logging.info('Reconstructing NASNetLarge and strictly restoring model and Adam')
    model,_=r.build_model()
    r.configure_stage(model,'normal_conv_1_1')
    optimizer=r.adam(1e-6)
    state=c.restore_checkpoint(model,optimizer,CHECKPOINT,sidecar['cache_catalog_sha256'],'production')
    assert state==saved['state']
    model_hashes=r.weight_hashes(model.weights)
    optimizer_hashes=r.weight_hashes(optimizer.variables)
    assert model_hashes==saved['model_tensor_sha256']
    assert optimizer_hashes==saved['optimizer_tensor_sha256']
    iterations=int(optimizer.iterations.numpy())
    assert iterations==saved['optimizer_iterations']==1167
    assert model_hashes==r.weight_hashes(model.weights) and optimizer_hashes==r.weight_hashes(optimizer.variables)
    assert snapshot()==before
    return dict(status='PASS',device=device,checkpoint_hashes=before,completed_epochs=14,
                boundary=state['boundary'],strict_restore=True,weights_exact=True,optimizer_state_exact=True,
                optimizer_iterations=iterations,optimizer_updates=0,training_batches=0,
                history_epochs=list(range(1,15)),selection=history['selection'],
                architecture=sidecar['model'],class_order=sidecar['class_order'],
                model_tensor_sha256=model_hashes,artifacts_unchanged=True)


def inference():
    import tensorflow as tf
    disable_training(tf)
    device=r.gpu_setup()
    saved=read(CHECKPOINT/'checkpoint.json')
    sidecar=read(RUN/'metadata.json')
    before=snapshot()
    logging.info('Loading existing epoch14.keras; verifying every weight against epoch_14_resume')
    model=tf.keras.models.load_model(RUN/'epoch14.keras',compile=False)
    r.validate_model(model)
    assert r.weight_hashes(model.weights)==saved['model_tensor_sha256']
    model.trainable=False
    weights=r.weight_hashes(model.weights)
    rows=r.read_manifest(Path('manifests/VAL1_manifest.csv'))
    _,internal=r.split_val1(rows)
    selected=[]
    for label in ('benign','tumour'):
        ranked=sorted((row for row in internal if row['class_name']==label),key=lambda row:r.rank_id(row['sample_id'],'Stage2I-readiness|42|'))
        selected.extend(ranked[:4])
    paths=[r.image_path(row,EXTRACTED) for row in selected]
    reference=r.normalizers(r.REFERENCE)
    timings=e.Timings()
    tf.config.experimental.reset_memory_stats('GPU:0')
    logging.info('Eight deterministic VAL1 internal-validation JPEGs; batch1, inference only')
    predictions,scores=e.infer_rows(model,selected,paths,reference,1,'/GPU:0',timings,r.LOG,8)
    first_memory=tf.config.experimental.get_memory_info('GPU:0')
    # Repeat on the same non-VAL2 images to test determinism, not choose settings.
    second,_=e.infer_rows(model,selected,paths,reference,1,'/GPU:0',e.Timings(),r.LOG,8)
    difference=float(np.max(np.abs(np.array(scores)-np.array([p['p_tumour'] for p in second]))))
    assert predictions==second and difference==0
    assert weights==r.weight_hashes(model.weights)
    assert snapshot()==before
    assert r.preprocessing_settings()=={k:v for k,v in sidecar['preprocessing'].items() if k not in ('resize','dtype','scale')}
    return dict(status='PASS',device=device,batch_size=1,allocator=os.environ.get('TF_GPU_ALLOCATOR','default'),
                sample_ids=[row['sample_id'] for row in selected],samples=8,source='VAL1 internal validation JPEGs',
                predictions=predictions,repeat_max_absolute_difference=difference,threshold_crossings=0,
                weights_and_BN_unchanged=True,export_matches_resume_weights=True,artifacts_unchanged=True,
                optimizer_updates=0,training_batches=0,peak_current_gpu_bytes=first_memory,
                timing=timings.report(8,sum(timings.seconds.values())) if hasattr(timings,'seconds') else str(timings.__dict__),
                batch_decision='Conservative batch1 fixed; no alternative batch selected from outcomes; default allocator sufficient in measured inference test')


def tests():
    import pytest
    results={}
    for name,paths in [('focused',['tests/test_stage2i_readiness.py','tests/test_reimplementation.py','tests/test_stage2h_readiness.py']),('full',[])]:
        log=REPORTS/f'stage2i_{name}_tests.log'
        base=f'/tmp/stage2i_tests_{name}_{time.time_ns()}'
        with log.open('w') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',base,*paths])
        results[name]=dict(exit_code=int(code),log=str(log))
        print(name,code,flush=True)
        assert code==0
    return dict(status='PASS',tests=results)


def report():
    restored=read(REPORTS/'stage2i_restore_verification.json')
    inferred=read(REPORTS/'stage2i_inference_verification.json')
    tested=read(REPORTS/'stage2i_tests_verification.json')
    assert all(v['status']=='PASS' and not v['VAL2_images_accessed'] for v in (restored,inferred,tested))
    assert snapshot()==restored['checkpoint_hashes'], 'Stage2H artifacts changed after tests'
    manifest=Path('manifests/VAL2_manifest.csv')
    rows=r.read_manifest(manifest,'VAL2')  # Manifest bytes only; never resolve image paths.
    counts=dict(Counter(row['class_name'] for row in rows))
    assert counts=={'benign':24471,'tumour':9632} and len(rows)==34103
    r.sha256_file(r.REFERENCE,r.REFERENCE_SHA)
    inputs=dict(model=restored['checkpoint_hashes'][str(RUN/'epoch14.keras')],
                metadata=restored['checkpoint_hashes'][str(RUN/'metadata.json')],
                manifest=r.sha256_file(manifest),stain_reference=r.REFERENCE_SHA)
    assert not Path('runs/stage2i_val2_native_epoch14').exists()
    logs={name:(REPORTS/f'stage2i_{name}_tests.log').read_text() for name in ('focused','full')}
    summaries={name:next(line for line in reversed(log.splitlines()) if ' passed' in line) for name,log in logs.items()}
    assert all('failed' not in value for value in summaries.values())
    git_status=(REPORTS/'stage2i_git_status.txt').read_text(encoding='utf-8-sig')
    diff_check=(REPORTS/'stage2i_diff_check.txt').read_text(encoding='utf-8-sig')
    assert diff_check.strip()=='PASS'
    command="""cd '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj'
env -u TF_GPU_ALLOCATOR \\
  /home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python -B -u \\
  -m modern_pca.evaluate_reimplementation \\
  --model runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras \\
  --metadata runs/stage2h_production_resume_epoch11_cuda_malloc_async/metadata.json \\
  --manifest manifests/VAL2_manifest.csv \\
  --extracted '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/zenodo/_3825933/extracted' \\
  --cohort VAL2 --final-evaluation \\
  --readiness reports/stage2i_readiness.json \\
  --stain-reference 4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg \\
  --batch-size 1 --device gpu \\
  --output runs/stage2i_val2_native_epoch14"""
    value=dict(status='PASS',created_at_utc=r.utc_now(),checkpoint=str(CHECKPOINT),
        frozen_contract=f.scientific_contract(),input_sha256=inputs,source_sha256=f.source_hashes(),
        software=r.capture_environment()['software'],environment=r.capture_environment(),
        checkpoint_hashes=restored['checkpoint_hashes'],history_sha256=restored['checkpoint_hashes'][str(RUN/'history.json')],
        restore_verification=restored,inference_verification=inferred,tests=summaries,
        manifest=dict(path=str(manifest),sha256=inputs['manifest'],samples=len(rows),counts=counts,image_data_opened=False),
        preprocessing_unchanged_from_training=True,checkpoint_and_export_identical_weights=True,
        final_command=command,final_command_executed=False,output='runs/stage2i_val2_native_epoch14',
        git_diff_check='PASS',git_status=git_status,
        verification_tool_sha256=r.sha256_file(Path(__file__)),
        files_changed_this_task=['modern_pca/evaluate_reimplementation.py','modern_pca/stage2i_contract.py',
                                 'tests/test_stage2i_readiness.py','tools/stage2i_readiness.py','reports/stage2i_*'],
        VAL2_images_accessed=False,production_or_final_evaluation_started=False,commit_push_performed=False,
        comparison_only=dict(accuracy=.967,precision=.927,recall=.957,f1=.942,roc_auc=.9918),
        caveat=f.LABEL)
    dependencies={}
    for package in ('Pillow','staintools','spams-bin','opencv-python','scikit-learn'):
        distribution=importlib.metadata.distribution(package)
        hashes={}
        for item in distribution.files or []:
            name=str(item).replace('\\','/')
            relevant=(package in ('staintools','spams-bin') and name.endswith(('.py','.so'))
                      or package=='Pillow' and (name=='PIL/Image.py' or name.startswith('PIL/_imaging') and name.endswith('.so'))
                      or package=='opencv-python' and name.startswith('cv2/') and name.endswith('.so')
                      or package=='scikit-learn' and name in ('sklearn/metrics/_ranking.py','sklearn/metrics/_classification.py'))
            path=Path(distribution.locate_file(item))
            if relevant and path.is_file(): hashes[name]=r.sha256_file(path)
        dependencies[package]=hashes
    value['relevant_installed_dependency_sha256']=dependencies
    r.write_json(REPORTS/'stage2i_readiness.json',value)
    # Verify the exact proposed arguments against the newly frozen report without running evaluation.
    from types import SimpleNamespace
    args=SimpleNamespace(model=RUN/'epoch14.keras',metadata=RUN/'metadata.json',manifest=manifest,
        stain_reference=r.REFERENCE,cohort='VAL2',batch_size=1,device='gpu',readiness=REPORTS/'stage2i_readiness.json')
    f.validate_readiness(args,read(args.metadata))
    lines=['# Stage 2I readiness: PASS','',f.LABEL,'',
        'No training or final VAL2 evaluation was started. VAL2 images remain unopened. Manifest metadata and bytes only were inspected. No commit or push. Explicit user authorization is required before executing the proposed final command.','',
        '## Frozen checkpoint','',
        f'Primary: `{CHECKPOINT}`. Strict restore consumed every checkpoint object; model/Adam tensor hashes exactly matched. Completed epoch14, boundary normal_conv_1_1, LR1e-6, Adam iteration1167. Verification performed zero optimizer updates and no training batches. All source artifact hashes were unchanged before/after both GPU checks.',
        'History contains epochs1..14 exactly once; selection is `none; fixed epoch14`. Internal metrics are provenance only, never model selection.','',
        'The evaluator loads the existing `epoch14.keras` with Keras load_model(compile=False). It does not load an epoch_14_resume directory directly. Every exported model tensor was checked against the strictly restored primary checkpoint; no re-export or Stage2H file edit was needed.','',
        '| Artifact | SHA-256 |','|---|---|']
    lines += [f'| `{name}` | `{digest}` |' for name,digest in restored['checkpoint_hashes'].items()]
    lines += ['', '## Evaluator audit and freeze','',
        'Primary CLI: python -m modern_pca.evaluate_reimplementation. Required flags: --model, --metadata, --manifest, --extracted, --output, --cohort {VAL1,VAL2}. Optional flags: --final-evaluation; --readiness (mandatory for VAL2); --stain-reference (frozen reference default); --batch-size {1,2,4} default1; --device {cpu,gpu} defaultgpu; --log-every default1000; --log-level {INFO,DEBUG} defaultINFO. Final readiness permits only batch1/GPU/default allocator.',
        'Architecture: NASNetLarge, 209812820 parameters, RGB350x350, Flatten -> Dense256/ReLU -> Dense2/softmax. Portable export contains architecture and weights; validate_model checks input/output shape, head and parameter count. Exact weight-hash linkage to the reconstructed NASNetLarge checkpoint establishes architecture provenance. Output0 benign; output1 tumour.',
        'Preprocessing is the unchanged frozen helper: source JPEG checksum verification -> PIL RGB (no EXIF transpose) -> Pillow LANCZOS350x350 -> brightness standardization -> StainTools Macenko -> float32 /255. Stain reference is fitted without an additional brightness transform, as in training. No preprocessing failures may be skipped.',
        'The original-authors evaluate_paper entry point is not invoked or changed. Existing pure preprocessing, metric, CSV, timing and device utilities are reused unchanged; its three-output model loader and evaluation workflow are not used.',
        'Decision: output[1] >0.5 is tumour; equality is benign. No probability rounding before decisions or AUC. Float32 model outputs are losslessly promoted to float64; CSV uses .17g for round-trip preservation.',
        'Confusion orientation [[TN,FP],[FN,TP]]. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall/sensitivity=TP/(TP+FN); specificity=TN/(TN+FP); F1=2TP/(2TP+FP+FN). AUC uses sklearn roc_auc_score on unrounded tumour scores. Undefined metrics use null and an explicit list.',
        'Canonical manifest order is preserved. Unique IDs, class counts, mapping, row indices and paths are validated; each row is preprocessed and inferred once, including the final partial batch. No sample-limit option, no warm-up images, no augmentation, no TTA/P8/C8, no ensemble, calibration, training, adaptation or threshold selection.',
        'Model is frozen and called with training=False. All weights, including BN moving state, are hashed before/after. Checkpoint/export and metadata file hashes are also checked. Output must be a new directory (exist_ok=False); a failed run has no completion marker.',
        'Outputs: predictions.csv (both probabilities, predicted code, truth, row index, sample ID); metrics.json; confusion_matrix.json; environment.json (frozen config/source/checkpoint/readiness hashes); timing_summary.json; run.log; run_complete.json with artifact hashes. Serial timings cover contract/manifest, startup, loading, reference, preprocessing, inference/device transfer, integrity, metrics and outputs. The final timing/completion self-reporting tail is excluded and labelled.',
        'Gaps fixed before VAL2: mandatory readiness binding for hashes/software/scientific contract; stronger sidecar checks; reusable/tested one-pass inference loop; explicit confusion artifact and required study label; float32/deterministic GPU setup; checkpoint file immutability checks.',
        '', '## Frozen configuration and identity','', '```json',json.dumps(value['frozen_contract'],indent=2),'```','',
        f'Manifest `{manifest}` SHA-256 `{inputs["manifest"]}`: 34103 rows, benign24471/tumour9632; no underlying image file was opened or hashed. Stain reference SHA-256 `{r.REFERENCE_SHA}`.','',
        'Software:', '```json',json.dumps(value['software'],indent=2),'```','',
        'Source hashes:', '```json',json.dumps(value['source_sha256'],indent=2),'```','',
        '## Non-VAL2 verification and inference batch','',
        f'Batch1 is frozen conservatively. Eight predetermined VAL1 internal-validation JPEGs (four/class) exercised the real decode/resize/stain/inference path twice. Max repeated probability difference={inferred["repeat_max_absolute_difference"]}; threshold crossings={inferred["threshold_crossings"]}; weights/BN unchanged. No alternative batch was selected from results.',
        f'GPU current/peak allocation after the first pass: `{inferred["peak_current_gpu_bytes"]}` bytes. GPU details: `{json.dumps(inferred["device"])}`. Default allocator passed; cuda_malloc_async is not required by this bounded inference evidence and is not in the frozen command. No training Adam or accumulator is allocated for final evaluation.',
        f'Focused: {summaries["focused"]}. Full: {summaries["full"]}. Synthetic tests cover threshold ties, unrounded AUC, CSV precision, confusion/metrics, once-only coverage, no augmentation, inference mode, BN/weight immutability, deterministic predictions, contract drift, fresh output and end-to-end output artifacts. GPU checks additionally prove actual final checkpoint restoration, export equivalence and unchanged files.',
        'git diff --check: PASS. Test logs and verification JSON files are alongside this report.','',
        '## Working tree','', 'Files changed in this task: '+', '.join(value['files_changed_this_task'])+'. Existing Stage2H modifications were preserved.',
        '```text',git_status,'```','',
        '## Predetermined descriptive comparison','',
        'Paper native VAL2 reference supplied for later descriptive comparison only: accuracy approximately96.7%, precision0.927, recall0.957, F1 0.942, AUC0.9918. No tuning, model selection, retraining or result-dependent rerun is authorized.',
        '', '## Proposed final command — NOT EXECUTED','', '```bash',command,'```','',
        'The output directory is currently absent. The readiness report freezes configuration, not permission to run it. Stop for explicit authorization.','']
    (REPORTS/'stage2i_readiness.md').write_text('\n'.join(lines))
    return dict(status='PASS',final_command_executed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['restore','inference','tests','report'])
    args=parser.parse_args()
    if args.phase=='tests':
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
    blocked=guard()
    logging.basicConfig(level=logging.INFO)
    value=globals()[args.phase]()
    assert not blocked
    value.update(VAL2_image_access_attempts=blocked,VAL2_images_accessed=False,
                 production_or_final_evaluation_started=False,completed_at_utc=r.utc_now())
    r.write_json(REPORTS/f'stage2i_{args.phase}_verification.json',value)
    print(json.dumps({key:value[key] for key in ('status','VAL2_images_accessed','production_or_final_evaluation_started')}))


if __name__=='__main__':
    main()
