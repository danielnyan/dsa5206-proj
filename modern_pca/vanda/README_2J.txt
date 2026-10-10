DSA5206 Stage 2J — Vanda 2 × A40, one SICAP epoch15

NEW files only:
 modern_pca/vanda_sicapv2_distributed.py
 vanda_smoke/vanda_sicapv2_epoch15.pbs

Requirements on Vanda:
 - Existing Vanda epoch14.keras SHA256 433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde
 - 9959 SICAP images in /scratch/e1536052/DSA5206/sicap_transfer/images
 - Selected manifest and original frozen stage2j_epoch15_plan.json in repository
 - Existing modern_pca.sicapv2 and preprocessing modules, Python venv with staintools/spams

1. Upload the two files into the new-file destinations above.
2. Validate syntax:
 python -m py_compile modern_pca/vanda_sicapv2_distributed.py
 bash -n vanda_smoke/vanda_sicapv2_epoch15.pbs
3. NO-UPDATE GPU PREFLIGHT (default):
 qsub -v MODE=preflight vanda_smoke/vanda_sicapv2_epoch15.pbs
 Check log for PREFLIGHT PASS; verify qstat -xf JOBID Exit_status=0.
4. Only after preflight passes, submit EXACTLY ONE real training run:
 qsub -v MODE=train vanda_smoke/vanda_sicapv2_epoch15.pbs
 IMPORTANT: train mode uses a fixed output directory and refuses to overwrite any contents.
5. Successful run outputs:
 /scratch/e1536052/DSA5206/vanda_runs/sicap_epoch15_vanda_2xa40/
  contract.json, epoch_15_resume/, epoch15.keras, history.json, metadata.json, run_complete.json

This is a secondary transfer experiment with 10x field-of-view deviation, NOT paper post-VAL1 reproduction.
The original laptop plan has physical batch 4 and its starting laptop model hash;
these two provenance changes are explicitly recorded in the new contract.json.
No VAL2 evaluation is executed here. Epoch15 VAL2 comparison needs a separate evaluation run.

Known limitations:
 - Syntax validation is local; no actual Vanda runtime test is possible here.
 - The per-image normalization happens serially in batches and may dominate runtime.
 - Preflight tests inference at 50/50 but not optimizer updates; the previously validated
   Vanda Stage2H distributed trainer supplies the train-step implementation.
