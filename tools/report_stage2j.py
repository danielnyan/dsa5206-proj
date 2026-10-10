"""Assemble the Stage 2J assessment from measured evidence; no training/evaluation."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess

from modern_pca import reimplementation as r, sicapv2 as s, train_sicapv2 as t
from modern_pca.stage2i_contract import source_hashes as primary_sources
from tools.audit_sicapv2 import guard
from tools.inspect_sicapv2 import ROOT


def read(path): return json.loads(Path(path).read_text())


def main():
    guard()
    archive=read('reports/stage2j_archive_audit.json')
    pre=read('reports/stage2j_preprocessing.json')
    gpu=read('reports/stage2j_gpu_verification.json')
    tests=read('reports/stage2j_tests_verification.json')
    primary=read('reports/stage2i_readiness.json')
    assert primary_sources()==primary['source_sha256'], 'Stage 2I frozen sources changed'
    assert archive['decoded_images']==archive['images']==archive['masks']==18783 and not archive['errors']
    assert pre['samples']==pre['successes']==archive['selected_samples']==9959 and not pre['failures']
    assert not archive['overlap']['VAL1']['count'] and not archive['overlap']['VAL2']['count']
    assert not archive['duplicate_raw_groups'] and not archive['duplicate_pixel_groups']
    assert gpu['status']==tests['status']=='PASS' and not gpu['blocked_VAL2_attempts'] and not tests['blocked_VAL2_attempts']
    with (ROOT/'audit/images.csv').open() as stream: rows=list(csv.DictReader(stream))
    selected=[row for row in rows if row['partition']=='train']
    assert {v['sample_id']:v['file_sha256'] for v in selected}==pre['selected_image_sha256']
    manifest=ROOT/'audit/selected_train.csv'
    r.sha256_file(manifest,archive['selected_manifest_sha256'])
    model=Path('runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras')
    r.sha256_file(model,t.MODEL_SHA)
    # Verify the existing protected artifacts against the prior read-only snapshot.
    restored=read('reports/stage2i_restore_verification.json')
    for name,digest in restored['checkpoint_hashes'].items(): r.sha256_file(Path(name),digest)
    audit_root=ROOT/'extracted/SICAPv2'
    global_labels=s.table((audit_root/'wsi_labels.xlsx').read_bytes(),['slide_id','patient_id','Gleason_primary','Gleason_secondary'])
    global_counts=dict(Counter(f"{v['Gleason_primary']}+{v['Gleason_secondary']}" for v in global_labels))
    fold_integrity={}
    by_name={row['filename']:row for row in rows}
    for number in range(1,5):
        parts=[]
        for split in ('Train','Test'):
            table=s.table((audit_root/f'partition/Validation/Val{number}/{split}.xlsx').read_bytes(),['image_name','NC','G3','G4','G5','G4C'])
            parts.append([by_name[v['image_name']] for v in table])
        fold_integrity[f'Val{number}']=s.patient_partition(*parts)
        assert {v['filename'] for values in parts for v in values}=={v['filename'] for v in selected}
    candidates={}
    for name,values in [('official_train_only',selected),('all_official_train_plus_test',[v for v in rows if v['partition']!='unlisted'])]:
        candidates[name]=dict(samples=len(values),benign=sum(v['ground_truth_code']=='0' for v in values),
            tumour=sum(v['ground_truth_code']=='1' for v in values),patients=len({v['patient_id'] for v in values}),slides=len({v['slide_id'] for v in values}))
    mask_cross={label:dict(Counter(v['majority_nonzero_value'] for v in rows if v['official_label']==label)) for label in ('NC','G3','G4','G5','unlisted')}
    quality=dict(all_white_images=sum(float(v['bright_pixel_fraction'])==1 for v in rows),
        constant_images=sum(float(v['std_rgb'])==0 for v in rows),
        max_bright_fraction=max(float(v['bright_pixel_fraction']) for v in rows),
        min_rgb_std=min(float(v['std_rgb']) for v in rows),
        mean_rgb_ranges={c:[min(float(v['mean_'+c]) for v in rows),max(float(v['mean_'+c]) for v in rows)] for c in ('r','g','b')},
        meaning='Brightness >=240 in all channels and RGB std are descriptive QC only; no exclusion thresholds.')
    quality['normalized_constant_images']=sum(v['std']==0 for v in pre['records'])
    quality['normalized_std_range']=[min(v['std'] for v in pre['records']),max(v['std'] for v in pre['records'])]
    quality['normalized_mean_rgb_ranges']=[[min(v['mean_rgb'][i] for v in pre['records']),max(v['mean_rgb'][i] for v in pre['records'])] for i in range(3)]
    count=len(selected); steps=math.ceil(count/100)
    seconds=sum(v['seconds'] for v in pre['records'])
    prior=read('reports/stage2h_epoch12_oom_investigation.json')['comparison'][0]
    compute_seconds=count/prior['steady_images_per_second'][-1]
    runtime=dict(preprocessing_measured_seconds=seconds,full_preprocessing_audit_seconds=pre['seconds'],
        historical_compiled_training_images_per_second=prior['steady_images_per_second'][-1],
        estimated_compute_seconds=compute_seconds,serial_preprocessing_plus_compute_seconds=seconds+compute_seconds,
        proposed_wall_clock_minutes=[35,55],optimizer_updates=steps,full_logical_batches=count//100,
        final_logical_batch_samples=count%100,physical_batches=math.ceil(count/4),
        previous_compiled_peak_bytes=prior['peak_bytes'],zero_update_probe=gpu['result']['memory_bytes'],
        cache_proposed=False,optional_uint8_cache_bytes=count*350*350*3,
        output_space_estimate_bytes=5_000_000_000,free_data_volume_bytes=shutil.disk_usage(ROOT).free,
        limitation='Estimate, not a timed production epoch. Eager diagnostic peak is close to current TensorFlow GPU budget; compiled Stage 2H same-boundary evidence used separately. Keep other GPU workloads closed.')
    comparison=dict(A='Stage 2I native fixed epoch14 VAL2 result',B='fixed epoch14 plus exactly one SICAP epoch15, then VAL2',
        metrics=['accuracy','precision','recall','sensitivity','specificity','f1','roc_auc','TN','FP','FN','TP'],
        probability_analysis='Per-image unrounded tumour probability delta B-A; crossings at p_tumour >0.5 (equality benign)',
        selection='No tuning, calibration, checkpoint selection, retry for VAL2 improvement or outcome-based sample choice',
        descriptive_paper_values={'native_VAL2':dict(accuracy=.967,precision=.927,recall=.957,f1=.942,roc_auc=.9918),
            'paper_VAL2_after_VAL1':dict(accuracy=.974,precision=.951,recall=.955,f1=.953,roc_auc=.9939)},
        caveat='User-provided approximate descriptive paper values. This is not reproduction of paper post-VAL1; report deterioration as deterioration.')
    plan=dict(status='CONDITIONAL GO',protocol=t.protocol(),samples=count,
        manifest_sha256=r.sha256_file(manifest),images_root=str((audit_root/'images').resolve()),
        preprocessing_failures=0,source_sha256=t.source_hashes(),software=r.capture_environment()['software'],
        comparison=comparison,condition='Explicit acceptance of 10x wider-field secondary domain/magnification deviation',
        starting_model_sha256=t.MODEL_SHA,archive_sha256=s.ARCHIVE_SHA)
    r.write_json(Path('reports/stage2j_epoch15_plan.json'),plan)
    t.training_rows(manifest,t.read_plan(Path('reports/stage2j_epoch15_plan.json')))
    train_output=Path('runs/stage2j_sicapv2_epoch15'); eval_output=Path('runs/stage2j_val2_epoch15')
    assert not train_output.exists() and not eval_output.exists()
    repo="/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj"
    python='/home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python'
    train_command=f"""cd '{repo}'
TF_GPU_ALLOCATOR=cuda_malloc_async {python} -B -u -m modern_pca.train_sicapv2 \\
  --plan reports/stage2j_epoch15_plan.json \\
  --model {model} \\
  --manifest '{manifest}' \\
  --images '{audit_root/'images'}' \\
  --output {train_output} \\
  --train-one-epoch --accept-magnification-deviation"""
    eval_command=f"""cd '{repo}'
env -u TF_GPU_ALLOCATOR {python} -B -u -m modern_pca.evaluate_sicapv2 \\
  --plan reports/stage2j_epoch15_plan.json \\
  --run {train_output} \\
  --baseline runs/stage2i_val2_native_epoch14 \\
  --manifest manifests/VAL2_manifest.csv \\
  --extracted '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/zenodo/_3825933/extracted' \\
  --output {eval_output} --final-evaluation"""
    # Use native Windows Git for its Windows checkout. Linux Git interprets the
    # existing CRLF working tree differently; do not normalize unrelated files.
    git_check=json.loads(Path('reports/stage2j_windows_git_check.json').read_text(encoding='utf-8-sig'))
    assert git_check['exit_code']==0 and not git_check['output']
    new_files=['modern_pca/sicapv2.py','modern_pca/train_sicapv2.py','modern_pca/evaluate_sicapv2.py',
        'tests/test_sicapv2.py','tests/test_sicapv2_training.py','tools/inspect_sicapv2.py','tools/audit_sicapv2.py',
        'tools/check_sicapv2_preprocessing.py','tools/verify_stage2j.py','tools/report_stage2j.py']
    for name in new_files:
        check=git_check['new_files'][name]
        assert check['exit_code'] in (0,1) and not check['output'],(name,check)
        r.sha256_file(Path(name),check['sha256'])
        assert all(line==line.rstrip() for line in Path(name).read_text().splitlines()),name
    report=dict(status='CONDITIONAL GO',created_at_utc=r.utc_now(),framing='SECONDARY external fine-tuning/transfer; not paper post-VAL1 reproduction',
        downloaded=True,exact_version_verified=True,archive_hashes_recorded=True,archive=archive,
        file_inventory_sha256=r.sha256_file(ROOT/'audit/archive_members.csv'),
        image_inventory_sha256=r.sha256_file(ROOT/'audit/images.csv'),candidate_mappings=candidates,
        selected='SICAPv2/partition/Test/Train.xlsx only; NC->0; G3/G4/G5->1',
        excluded=dict(official_test=2122,unlisted=6702,total=8824,ambiguous_official_primary_labels=0,selected_quality_exclusions=0),
        patient_partition_preserved=True,cross_validation_integrity=fold_integrity,global_Gleason_counts=global_counts,
        masks=dict(raw_histogram=archive['mask_pixel_counts'],majority_nonzero_by_official_label=mask_cross,
            interpretation='Actual values are exactly 0/3/4/5, empirically associated with NC/G3/G4/G5; the stale Version1 readme says 0/1/2/3. Six official G3/G4 labels disagree with mask majority grade but remain tumour under either binary mapping. No mask-derived training labels or thresholds. Zero does not distinguish benign tissue from unannotated/background; numeric mixtures do not independently certify biological mixtures.',
            mixed_zero_nonzero_patches=archive['mixed_zero_nonzero'],multiple_nonzero_value_patches=archive['multiple_nonzero_values']),
        magnification=dict(released='512x512 RGB patches documented at10x, 50% overlap',source_scanner='Authors describe40x Ventana iScanCoreo acquisition',
            mpp=None,physical_FOV_microns=None,available_high_resolution_WSI=False,
            primary_source='MAIN.py p_s600/m_p_s350 and wsi_process.py comment40x, level0 read_region',
            nominal_linear_FOV_ratio=(512/10)/(600/40),nominal_area_ratio=((512/10)/(600/40))**2,
            caveat='Ratios assume comparable scanner calibration; MPP is not established. Whole-patch resize preserves a much wider field. Interpolation cannot recover optical resolution. Crop/rescale would change labels/context and is not proposed.'),
        quality=quality,preprocessing=dict(samples=pre['samples'],successes=pre['successes'],failures=pre['failures'],failure_rate=pre['failure_rate'],
            reference_sha256=r.REFERENCE_SHA,visual_review='Eight deterministic official training samples, two per grade, raw/normalized pairs. Stronger contrast and slight background speckling visible; no gross structural loss seen at preview scale. Not a histopathologist validation or exhaustive visual review.',
            report='reports/stage2j_preprocessing.json',contact_sheet='reports/stage2j_preprocessing_contact_sheet.png'),
        protocol=t.protocol(),epoch15_configuration_frozen=True,plan_sha256=r.sha256_file(Path('reports/stage2j_epoch15_plan.json')),
        runtime_memory=runtime,GPU_feasibility='PASS (zero-update real model probe plus existing compiled training evidence; no new optimizer updates)',
        tests=tests['result'],git_diff_check='PASS; additional trailing-whitespace check of all new code passed',
        primary_sources_unchanged=True,primary_checkpoints_history_unchanged=True,Stage2I_outcomes_inspected=False,
        training_started=False,new_VAL2_evaluation_started=False,VAL2_images_accessed=False,commit_push=False,
        comparison=comparison,files_added=new_files,training_command=train_command,post_training_VAL2_command=eval_command,
        sources=[s.DATASET_URL,'https://arxiv.org/html/2105.10490v1','4_WSI_pipeline/WSI_pipeline_v6/MAIN.py','4_WSI_pipeline/WSI_pipeline_v6/wsi_process.py'])
    r.write_json(Path('reports/stage2j_sicapv2_assessment.json'),report)
    a=candidates['official_train_only']; b=candidates['all_official_train_plus_test']
    md=f'''# Stage 2J — SICAPv2 one-epoch external post-training feasibility

**CONDITIONAL GO** for one predefined SECONDARY external fine-tuning experiment, conditional on accepting the documented 10× magnification/domain deviation. This is **not** reproduction of Tolkach's original-model → VAL1 post-training procedure. No improvement on VAL2 is promised.

## Acquisition and provenance

Official [Version 2, DOI 10.17632/9xxm58dvs3.2]({s.DATASET_URL}), published 22 October 2020, CC BY 4.0. Downloaded outside Git to `{ROOT}/SICAPv2.zip`.

- Exact download: {s.DOWNLOAD_URL}
- Archive bytes: {s.ARCHIVE_BYTES:,}; SHA256: `{s.ARCHIVE_SHA}` (matches official API record).
- {archive['archive_files']:,} files; {archive['archive_uncompressed_bytes']:,} uncompressed bytes.
- {archive['images']:,} JPEG images, {archive['masks']:,} PNG masks, 21 XLSX files, readme.txt and Thumbs.db. No higher-resolution WSI files.
- Extraction: `{audit_root}`. Every extracted file hash is recorded in `{ROOT}/audit/archive_members.csv`; image/mask details in `images.csv` beside it.
- Image properties: `{archive['image_properties']}`. Mask properties: `{archive['mask_properties']}`.
- {archive['slides']} slide identifiers and {archive['patients']} patient identifiers are explicitly provided by `wsi_labels.xlsx`, with primary/secondary global Gleason grades.
- Bundled readme has a stale “Version 1: 2020-05” header. Official version, archive hash and observed contents take precedence; this inconsistency is retained in the evidence.

## Labels and partition selection

Use **only `partition/Test/Train.xlsx`**, the main official training partition (the parent directory name “Test” denotes the held-out partition scheme). `NC→benign0`; `G3/G4/G5→tumour1`. `G4C` is ancillary cribriform information, not a fifth mutually exclusive class. All primary labels are one-hot and agree across supplied classification tables.

| Candidate | Samples | Benign | Tumour | Patients | Slides |
|---|---:|---:|---:|---:|---:|
| Recommended official train | {a['samples']} | {a['benign']} | {a['tumour']} | {a['patients']} | {a['slides']} |
| All official train + test (not selected) | {b['samples']} | {b['benign']} | {b['tumour']} | {b['patients']} | {b['slides']} |

Exclude **8,824** released images from this experiment: **2,122 official test** plus **6,702 not listed in the main classification train/test tables**. There are **0 ambiguous official primary labels** and **0 preprocessing/corruption exclusions** from the selected 9,959. Unlisted masks are not converted into new training labels. Do not enlarge training with the held-out test. Main train/test patient overlap is zero; all four supplied internal CV folds are also patient-disjoint and partition the main training set. Exact fold class/patient/slide counts are in the JSON.

Tumour prevalence in the selected set is {a['tumour']/a['samples']:.2%}. Use no class weighting or resampling. Keep mixed morphology patches with valid official primary labels; the patch classification task does not require every pixel to share that label.

## Mask interpretation and quality

**Actual mask values are exactly0/3/4/5, disagreeing with the readme's0/1/2/3 legend.** Literal global histogram: `{archive['mask_pixel_counts']}`. Majority-nonzero versus official labels: `{mask_cross}`. The observed association supports0=NC/unannotated/background and3/4/5=G3/G4/G5. Six official G3/G4 labels disagree with mask majority grade (four G4→majority3; two G3→majority4), but all remain **tumour** under the proposed binary mapping; retain official labels. Do not silently relabel or threshold masks. Zero does not separately encode tissue versus background. Multiple nonzero values show mixed annotations but cannot independently certify biological mixtures or distinguish boundary effects. Numeric mixtures: {archive['mixed_zero_nonzero']:,} zero/nonzero and {archive['multiple_nonzero_values']:,} multiple-nonzero-value patches. Of6,702 unlisted images,395 have nonzero annotations and6,307 have all-zero masks; neither group is added to training.

All {archive['images']:,} image/mask pairs decode and match dimensions; **0 corrupt/unreadable pairs**, **0 raw-byte duplicate groups**, **0 decoded-RGB duplicate groups**. Exact JPEG hash overlaps: **VAL1=0; VAL2=0**, compared against authenticated existing manifests only. No VAL2 images opened. Hash checks do not rule out re-encoded/near duplicates or establish cross-dataset patient identity. Within-slide 50% patch overlap creates expected spatial correlation; patient partitions are preserved.

QC: {quality}. These are descriptive values, not exclusion thresholds.

## Scale and preprocessing

The released readme documents 512×512 patches at10× with50% overlap. The [authors' paper](https://arxiv.org/html/2105.10490v1) describes40× source scanning and10× patch generation. No calibrated MPP or physical field of view is established by the released JPEG metadata, and this archive contains no higher-resolution WSI source. The primary repository pipeline uses600px at40×, resized to350 (`MAIN.py:12–14`, `wsi_process.py:35–53`). Under comparable calibration, SICAP's whole-patch field is approximately **3.41× wider /11.65× the area**. These are nominal ratios, not measured micrometre values.

The proposed whole512→350 resize therefore retains a substantially different morphology scale. **Interpolation cannot restore absent optical detail.** A crop or alternative resize would define another protocol and could invalidate whole-patch labels; neither is proposed.

**9,959/9,959 selected images passed** the unchanged PIL RGB → LANCZOS350 → frozen brightness standardizer → StainTools Macenko → float32/255 path; **0 failures (0%)**. Same reference SHA `{r.REFERENCE_SHA}`. Full measured preprocessing audit: {pre['seconds']/60:.2f}min. No cache was built. [Raw/normalized contact sheet](stage2j_preprocessing_contact_sheet.png): eight deterministic samples, two per official grade. Stronger contrast and slight background speckling are visible; no gross structural loss seen at preview scale. This is not exhaustive visual or histopathologist validation.

## Frozen epoch15 and feasibility

Start `{model}` (SHA `{t.MODEL_SHA}`), exactly one extra epoch; final boundary `normal_conv_1_1`; frozen/inference BN; fresh Adam LR1e-6, beta1=.9, beta2=.999, epsilon1e-7; no decay/clipping/AMSGrad; unweighted sparse categorical cross-entropy; physical4/effective100; same stateless horizontal/vertical flips atp=.5, seed42; float32; TF32off; cuda_malloc_async. Deterministic external manifest order and epoch15 flip key are frozen. Every selected image appears once. No SICAP test validation, checkpoint selection, VAL2-based adaptation, or14-epoch schedule restart.

**100 optimizer updates**: 99 groups of100 then59; 2,490 physical batches (last3). No updates were executed during this assessment. GPU visibility and a read-only4-image forward/backward probe passed with epoch14 weights, Adam slots and accumulation buffers allocated. Peak **{gpu['result']['memory_bytes']['peak']/1e9:.3f}GB**, current **{gpu['result']['memory_bytes']['current']/1e9:.3f}GB**. Weights, BN, Adam and buffers unchanged. Eager peak is close to the current available GPU budget; this is not a full-epoch guarantee. Prior same-boundary compiled Stage2H evidence peaked at5.857GB and processed13.392images/s. Keep other GPU workloads closed.

Serial preprocessing plus historical compute estimate: {(seconds+compute_seconds)/60:.1f}min; allow **35–55min** including tracing, I/O and checkpoint/export. Hardware contention may increase this. No large cache needed; optional uint8 tensors alone would be {count*350*350*3/1e9:.3f}GB. Reserve approximately5GB for the secondary model/Adam checkpoint/export/logs. Available data-volume space: {runtime['free_data_volume_bytes']/1e9:.1f}GB.

The secondary trainer saves `epoch_15_resume/`, `epoch15.keras`, standalone `history.json`, `metadata.json`, `run.log`, and `run_complete.json` into a fresh directory. Primary epoch1–14 history/checkpoints remain unchanged. It has no mid-epoch resume feature; an interrupted epoch is incomplete and must not be evaluated. Never rerun in response to VAL2 results.

## Comparison frozen before training

A = Stage2I native epoch14 VAL2; B = one SICAP epoch15 then VAL2. Compare accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC and TN/FP/FN/TP; save each image's unrounded tumour-probability shift B−A and threshold crossings. Fixed rule p_tumour>0.5, ties benign; batch1 GPU inference, no augmentation/TTA/calibration. Do not inspect VAL2 errors to select samples, LR, threshold, epochs, preprocessing or repeat/select runs. Report worse outcomes as worse.

User-provided approximate descriptive paper values only: native .967/.927/.957/.942/.9918; paper post-VAL1 .974/.951/.955/.953/.9939 (accuracy/precision/recall/F1/AUC). Our SICAP sequence is scientifically different. These numbers are not targets. Stage2I outcome artifacts were not inspected during this assessment.

## Validation and boundaries

- Focused tests: {tests['result']['focused']['summary']}.
- Full tests: {tests['result']['full']['summary']}.
- `git diff --check`: PASS; new-file trailing whitespace check: PASS.
- Primary frozen source hashes and checkpoint/history hashes: unchanged.
- Training started: **NO**. New VAL2 evaluation: **NO**. VAL2 image access: **NO**. Commit/push: **NO**.
- Added code: {', '.join('`'+p+'`' for p in new_files)}. Reports/evidence use only `reports/stage2j_*`; dataset stays outside Git.

## Proposed commands — NOT EXECUTED

Training, only if the10× secondary deviation is accepted:

```bash
{train_command}
```

After successful epoch15 and an authenticated completed Stage2I baseline, one fixed secondary VAL2 evaluation:

```bash
{eval_command}
```

The evaluator verifies the secondary completion marker/model hash and baseline identity, uses the same frozen inference/metric helpers, and writes separate outputs. It does not relax Stage2I's epoch14-only contract. Source, software or plan changes require revalidation; these commands fail closed on drift.
'''
    Path('reports/stage2j_sicapv2_assessment.md').write_text(md,encoding='utf-8')
    Path('reports/stage2j_diff_check.txt').write_text('PASS\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],selected=a,tests=report['tests'],runtime=runtime),indent=2))


if __name__=='__main__': main()
