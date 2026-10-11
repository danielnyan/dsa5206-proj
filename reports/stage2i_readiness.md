# Stage 2I readiness: PASS

Constrained binary reimplementation trained on released VAL1 and evaluated on held-out VAL2; not an exact reproduction of Tolkach et al.

No training or final VAL2 evaluation was started. VAL2 images remain unopened. Manifest metadata and bytes only were inspected. No commit or push. Explicit user authorization is required before executing the proposed final command.

## Frozen checkpoint

Primary: `runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch_14_resume`. Strict restore consumed every checkpoint object; model/Adam tensor hashes exactly matched. Completed epoch14, boundary normal_conv_1_1, LR1e-6, Adam iteration1167. Verification performed zero optimizer updates and no training batches. All source artifact hashes were unchanged before/after both GPU checks.
History contains epochs1..14 exactly once; selection is `none; fixed epoch14`. Internal metrics are provenance only, never model selection.

The evaluator loads the existing `epoch14.keras` with Keras load_model(compile=False). It does not load an epoch_14_resume directory directly. Every exported model tensor was checked against the strictly restored primary checkpoint; no re-export or Stage2H file edit was needed.

| Artifact | SHA-256 |
|---|---|
| `runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras` | `5cecd53138d74cffc7313521a1c78544096d7e872440b496631a5666534e1e61` |
| `runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch_14_resume/checkpoint.json` | `013d86025ecded045370dbf3008cd841a0598e3d2e8fc87f7080e66f6cbc1c1c` |
| `runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch_14_resume/state.data-00000-of-00001` | `50d8ce7b94d51d5412b61d51d8d63c2c02f6e046b7af15d2bbf30bedf8f0b733` |
| `runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch_14_resume/state.index` | `9016df46969a636a9b26da9f8cc4b83d4adff1d49eb51d412dec242395cf7457` |
| `runs/stage2h_production_resume_epoch11_cuda_malloc_async/history.json` | `e7bb636e9e8725148c4e6124b3617bab13a137d23343457bcc4c7f7091dc4b83` |
| `runs/stage2h_production_resume_epoch11_cuda_malloc_async/metadata.json` | `9cc2f0542642369555511cc3ddb5e4ed870343ee39d34b0122a86277fd568602` |

## Evaluator audit and freeze

Primary CLI: python -m modern_pca.evaluate_reimplementation. Required flags: --model, --metadata, --manifest, --extracted, --output, --cohort {VAL1,VAL2}. Optional flags: --final-evaluation; --readiness (mandatory for VAL2); --stain-reference (frozen reference default); --batch-size {1,2,4} default1; --device {cpu,gpu} defaultgpu; --log-every default1000; --log-level {INFO,DEBUG} defaultINFO. Final readiness permits only batch1/GPU/default allocator.
Architecture: NASNetLarge, 209812820 parameters, RGB350x350, Flatten -> Dense256/ReLU -> Dense2/softmax. Portable export contains architecture and weights; validate_model checks input/output shape, head and parameter count. Exact weight-hash linkage to the reconstructed NASNetLarge checkpoint establishes architecture provenance. Output0 benign; output1 tumour.
Preprocessing is the unchanged frozen helper: source JPEG checksum verification -> PIL RGB (no EXIF transpose) -> Pillow LANCZOS350x350 -> brightness standardization -> StainTools Macenko -> float32 /255. Stain reference is fitted without an additional brightness transform, as in training. No preprocessing failures may be skipped.
The original-authors evaluate_paper entry point is not invoked or changed. Existing pure preprocessing, metric, CSV, timing and device utilities are reused unchanged; its three-output model loader and evaluation workflow are not used.
Decision: output[1] >0.5 is tumour; equality is benign. No probability rounding before decisions or AUC. Float32 model outputs are losslessly promoted to float64; CSV uses .17g for round-trip preservation.
Confusion orientation [[TN,FP],[FN,TP]]. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall/sensitivity=TP/(TP+FN); specificity=TN/(TN+FP); F1=2TP/(2TP+FP+FN). AUC uses sklearn roc_auc_score on unrounded tumour scores. Undefined metrics use null and an explicit list.
Canonical manifest order is preserved. Unique IDs, class counts, mapping, row indices and paths are validated; each row is preprocessed and inferred once, including the final partial batch. No sample-limit option, no warm-up images, no augmentation, no TTA/P8/C8, no ensemble, calibration, training, adaptation or threshold selection.
Model is frozen and called with training=False. All weights, including BN moving state, are hashed before/after. Checkpoint/export and metadata file hashes are also checked. Output must be a new directory (exist_ok=False); a failed run has no completion marker.
Outputs: predictions.csv (both probabilities, predicted code, truth, row index, sample ID); metrics.json; confusion_matrix.json; environment.json (frozen config/source/checkpoint/readiness hashes); timing_summary.json; run.log; run_complete.json with artifact hashes. Serial timings cover contract/manifest, startup, loading, reference, preprocessing, inference/device transfer, integrity, metrics and outputs. The final timing/completion self-reporting tail is excluded and labelled.
Gaps fixed before VAL2: mandatory readiness binding for hashes/software/scientific contract; stronger sidecar checks; reusable/tested one-pass inference loop; explicit confusion artifact and required study label; float32/deterministic GPU setup; checkpoint file immutability checks.

## Frozen configuration and identity

```json
{
  "model": "NASNetLarge",
  "input_shape": [
    350,
    350,
    3
  ],
  "head": [
    "Flatten",
    "Dense(256,relu)",
    "Dense(2,softmax)"
  ],
  "parameters": 209812820,
  "class_order": [
    "benign",
    "tumour"
  ],
  "selection": "none; fixed epoch14",
  "threshold": 0.5,
  "decision": "p_tumour > 0.5; equality benign",
  "precision": "float32",
  "TF32": false,
  "training": false,
  "augmentation": false,
  "TTA": false,
  "calibration": false,
  "warmup_samples": 0,
  "batch_size": 1,
  "device": "gpu",
  "allocator": "default",
  "manifest_sha256": "7b95ac02798e5971fc2722d3c37e8e7613bc1d020dee9a1a5d49ac939cf746d9",
  "counts": {
    "benign": 24471,
    "tumour": 9632
  },
  "samples": 34103,
  "stain_reference_sha256": "1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1",
  "preprocessing": {
    "source_decode": "PIL RGB; no EXIF transpose",
    "resize_before_normalization": true,
    "brightness": "legacy StainTools brightness/luminosity standardization; 95th-percentile default; API-compatible implementation",
    "stain_method": "macenko",
    "stain_options": "installed defaults bound to backend source hashes",
    "reference_fit": "staintools.read_image; no brightness transform",
    "failure_policy": "abort",
    "backend_source_sha256": {
      "staintools": "52a377f0c9eafc921367536b58131b8d2406afe23e3b38f3033e5666e76dfdfa",
      "opencv-python": "1a06287c8d0d6e1e7c8d1c57d3ccef223bfc777a8566eba1315dff09308b4527",
      "spams": "unavailable"
    },
    "resize": "Pillow LANCZOS 350x350",
    "dtype": "float32",
    "scale": "1/255"
  },
  "metrics": [
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc"
  ],
  "confusion_matrix": "[[TN,FP],[FN,TP]]",
  "probabilities": "unrounded float32 losslessly promoted to float64; CSV round-trip precision",
  "description": "Constrained binary reimplementation trained on released VAL1 and evaluated on held-out VAL2; not an exact reproduction of Tolkach et al."
}
```

Manifest `manifests/VAL2_manifest.csv` SHA-256 `7b95ac02798e5971fc2722d3c37e8e7613bc1d020dee9a1a5d49ac939cf746d9`: 34103 rows, benign24471/tumour9632; no underlying image file was opened or hashed. Stain reference SHA-256 `1d31d575c21dcbcf172957b0783c37fcb07872d3129b3a7b3e86ede25ea0a1c1`.

Software:
```json
{
  "python": "3.11.16",
  "tensorflow": "2.20.0",
  "keras": "3.15.1",
  "numpy": "1.26.4",
  "scikit-learn": "1.5.2",
  "Pillow": "10.4.0",
  "staintools": "2.1.2",
  "h5py": "3.16.0",
  "spams": null,
  "spams-bin": "2.6.13",
  "opencv-python": "4.11.0.86",
  "opencv-python-headless": null
}
```

Source hashes:
```json
{
  "modern_pca/evaluate_reimplementation.py": "8ce23129b7589fdf4761dbe40412b5f34ebea818c738b65e95dd9e44b911550c",
  "modern_pca/stage2i_contract.py": "ebe50dae39c1065e82a5e853cbfb2ea329e9242610f2948f6aa35c632cd94e21",
  "modern_pca/reimplementation.py": "851b807fdc6d95f5880a4ccf51e0182087931df1fb4c42fa7a6984369cc46550",
  "modern_pca/models.py": "9b52264bd63b2fe71ed9cac46adace1ed0eb4ebebce0c05bfb2f43d55164758f",
  "modern_pca/evaluate_paper.py": "e79d883bab30061dce3b68dc493a454c8a5b5f763e75916d2f31485345654c8b",
  "modern_pca/cached_training.py": "a9bfaafd350f4b96f58cae6a3a4297e71ff5a440089956bb3379a5e82416351d",
  "tools/build_validation_manifests.py": "9b9d6e35716138c4cc25a0d3547041d92c09c1dad61bc4ad0e3fe5bd4b2a63cb"
}
```

## Non-VAL2 verification and inference batch

Batch1 is frozen conservatively. Eight predetermined VAL1 internal-validation JPEGs (four/class) exercised the real decode/resize/stain/inference path twice. Max repeated probability difference=0.0; threshold crossings=0; weights/BN unchanged. No alternative batch was selected from results.
GPU current/peak allocation after the first pass: `{'current': 861226496, 'peak': 2136091648}` bytes. GPU details: `{"build": {"cpu_compiler": "clang 18", "cuda_compute_capabilities": ["sm_60", "sm_70", "sm_80", "sm_89", "compute_90"], "cuda_version": "12.5.1", "cudnn_version": "9", "is_cuda_build": true, "is_rocm_build": false, "is_tensorrt_build": false}, "cuda_visible_devices": "unset", "deterministic_ops": true, "gpu_details": [{"compute_capability": [8, 9], "device_name": "NVIDIA GeForce RTX 4080 Laptop GPU"}], "keras": "3.15.1", "synchronous_execution": true, "tensor_float_32": false, "tensorflow": "2.20.0", "visible_gpus": ["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]}`. Default allocator passed; cuda_malloc_async is not required by this bounded inference evidence and is not in the frozen command. No training Adam or accumulator is allocated for final evaluation.
Focused: 86 passed in 11.30s. Full: 279 passed in 55.63s. Synthetic tests cover threshold ties, unrounded AUC, CSV precision, confusion/metrics, once-only coverage, no augmentation, inference mode, BN/weight immutability, deterministic predictions, contract drift, fresh output and end-to-end output artifacts. GPU checks additionally prove actual final checkpoint restoration, export equivalence and unchanged files.
git diff --check: PASS. Test logs and verification JSON files are alongside this report.

## Working tree

Files changed in this task: modern_pca/evaluate_reimplementation.py, modern_pca/stage2i_contract.py, tests/test_stage2i_readiness.py, tools/stage2i_readiness.py, reports/stage2i_*. Existing Stage2H modifications were preserved.
```text
 M modern_pca/cached_training.py
 M modern_pca/evaluate_reimplementation.py
 M modern_pca/train_reimplementation.py
 M tests/test_stage2h_readiness.py
?? modern_pca/stage2i_contract.py
?? reports/stage2e_report_generation.log
?? reports/stage2e_tests.log
?? reports/stage2f_diagnostic_tests.log
?? reports/stage2f_report_generation.log
?? reports/stage2f_tests.log
?? reports/stage2g_cache_generation.log
?? reports/stage2g_focused_tests.log
?? reports/stage2g_tests.log
?? reports/stage2h_epoch12_focused_tests.log
?? reports/stage2h_epoch12_full_tests.log
?? reports/stage2h_epoch12_git_status.txt
?? reports/stage2h_epoch12_oom_investigation.json
?? reports/stage2h_epoch12_oom_investigation.md
?? reports/stage2h_focused_tests.log
?? reports/stage2h_tests.log
?? reports/stage2h_validation_batch2_verification.json
?? reports/stage2h_validation_batch2_verification.log
?? reports/stage2h_validation_batch_verification.json
?? reports/stage2h_validation_batch_verification.log
?? reports/stage2h_validation_focused_tests.log
?? reports/stage2h_validation_full_tests.log
?? reports/stage2i_diff_check.txt
?? reports/stage2i_focused_tests.log
?? reports/stage2i_full_tests.log
?? reports/stage2i_git_status.txt
?? reports/stage2i_inference_verification.json
?? reports/stage2i_readiness.json
?? reports/stage2i_readiness.md
?? reports/stage2i_report_verification.json
?? reports/stage2i_restore_verification.json
?? reports/stage2i_tests_verification.json
?? tests/test_stage2h_oom_candidates.py
?? tests/test_stage2i_readiness.py
?? tools/report_stage2h_epoch12.py
?? tools/stage2h_epoch12_probe.py
?? tools/stage2i_readiness.py
?? tools/verify_stage2h_validation.py

```

## Predetermined descriptive comparison

Paper native VAL2 reference supplied for later descriptive comparison only: accuracy approximately96.7%, precision0.927, recall0.957, F1 0.942, AUC0.9918. No tuning, model selection, retraining or result-dependent rerun is authorized.

## Proposed final command — NOT EXECUTED

```bash
cd '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/GitHub/dsa5206-proj'
env -u TF_GPU_ALLOCATOR \
  /home/soonh/miniconda3/envs/dsa5206-reimplementation/bin/python -B -u \
  -m modern_pca.evaluate_reimplementation \
  --model runs/stage2h_production_resume_epoch11_cuda_malloc_async/epoch14.keras \
  --metadata runs/stage2h_production_resume_epoch11_cuda_malloc_async/metadata.json \
  --manifest manifests/VAL2_manifest.csv \
  --extracted '/mnt/d/DSML 2025/Semester 1 (2026)/DSA5206/Project/data/zenodo/_3825933/extracted' \
  --cohort VAL2 --final-evaluation \
  --readiness reports/stage2i_readiness.json \
  --stain-reference 4_WSI_pipeline/WSI_pipeline_v6/images/standard_he_stain_small.jpg \
  --batch-size 1 --device gpu \
  --output runs/stage2i_val2_native_epoch14
```

The output directory is currently absent. The readiness report freezes configuration, not permission to run it. Stop for explicit authorization.
