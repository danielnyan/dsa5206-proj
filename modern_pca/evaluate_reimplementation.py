"""Frozen binary evaluation for the constrained reimplementation only.

VAL2 requires explicit --final-evaluation and an epoch-14 production sidecar.
Scientific outputs and performance timings are stored separately. No fitting,
augmentation, threshold selection, calibration, or checkpoint selection occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from . import reimplementation as r
from . import stage2i_contract as frozen
from .evaluate_paper import (binary_decision, calculate_binary_metrics, csv_bytes,
                             Timings, Progress, select_device)


def probabilities_2(values, count):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (count,2) or not np.isfinite(values).all() or
        np.any(values < 0) or np.any(values > 1) or
        np.any(np.abs(values.sum(axis=1)-1) > 1e-5)):
        raise ValueError("Invalid two-class softmax probabilities")
    return values


def validate_sidecar(sidecar, model, final_evaluation):
    if sidecar.get("class_order") != ["benign","tumour"]:
        raise ValueError("Require benign/tumour class order")
    if sidecar.get("stain_reference_sha256") != r.REFERENCE_SHA:
        raise ValueError("Stain reference provenance mismatch")
    if sidecar.get("model_sha256") != r.sha256_file(model):
        raise ValueError("Checkpoint checksum mismatch")
    if final_evaluation and (sidecar.get("purpose") != "production_epoch14" or sidecar.get("completed_epochs") != 14):
        raise ValueError("VAL2 requires the fixed epoch-14 production checkpoint; smoke checkpoints forbidden")
    if final_evaluation:
        contract = frozen.scientific_contract()
        expected_model = dict(backbone='NASNetLarge', weights='imagenet', input=[350,350,3],
                              head=contract['head'], parameters=r.PARAMETERS)
        if (sidecar.get('model') != expected_model or sidecar.get('precision') != 'float32'
                or sidecar.get('preprocessing') != contract['preprocessing']):
            raise ValueError('Checkpoint architecture/precision/preprocessing contract mismatch')


def infer_rows(model, rows, paths, reference, batch_size, device, timings, logger, log_every):
    """Exactly one ordered inference per row; no warm-up or augmentation."""
    import tensorflow as tf
    if (not rows or len(paths) != len(rows) or batch_size not in (1,2,4)
            or len({row['sample_id'] for row in rows}) != len(rows)
            or any(row['ground_truth_code'] != r.CLASSES.get(row['class_name']) for row in rows)):
        raise ValueError('Invalid/duplicate evaluation rows or class mapping')
    predictions, scores = [], []
    progress = Progress(len(rows),log_every,30,logger)
    for offset in range(0,len(rows),batch_size):
        batch = rows[offset:offset+batch_size]
        with timings.measure('preprocessing'):
            inputs = np.stack([r.preprocess_legacy(row,path,reference)
                               for row,path in zip(batch,paths[offset:offset+batch_size])])
        with timings.measure('inference'):
            with tf.device(device):
                probabilities = probabilities_2(model(inputs,training=False).numpy(),len(batch))
        for row, values in zip(batch,probabilities):
            score = float(values[1])
            scores.append(score)
            predictions.append(dict(row_index=row['row_index'],sample_id=row['sample_id'],
                ground_truth_code=row['ground_truth_code'],p_benign=float(values[0]),
                p_tumour=score,predicted_binary_code=int(binary_decision(score))))
        progress.update(min(offset+batch_size,len(rows)))
    if [p['sample_id'] for p in predictions] != [row['sample_id'] for row in rows]:
        raise ValueError('Evaluation coverage/order mismatch')
    return predictions, scores


def run(args):
    if args.cohort == "VAL2" and not args.final_evaluation:
        raise ValueError("VAL2 is locked until explicit final evaluation")
    start = time.perf_counter()
    timings = Timings()
    with timings.measure('manifest'):
        rows = r.read_manifest(args.manifest,args.cohort)
        sidecar = json.loads(args.metadata.read_text())
        metadata_sha = r.sha256_file(args.metadata)
        validate_sidecar(sidecar,args.model,args.final_evaluation)
        readiness = frozen.validate_readiness(args,sidecar) if args.cohort == 'VAL2' else None
    r.ensure_output_separation(args.extracted,[args.output])
    logger = r.logging_setup(args.output,args.log_level)
    with timings.measure("startup"):
        import tensorflow as tf
        tf.config.experimental.set_synchronous_execution(True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.keras.mixed_precision.set_global_policy('float32')
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        if args.device == 'gpu':
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu,True)
        device, device_info = select_device(tf,args.device)
    with timings.measure("manifest"):
        paths = [r.image_path(row,args.extracted,final_evaluation=args.final_evaluation) for row in rows]
    logger.info("Manifest verified: %s images=%d",args.cohort,len(rows))
    with timings.measure("model_load"):
        logger.info("Loading frozen two-output model")
        with tf.device(device):
            model = tf.keras.models.load_model(args.model,compile=False)
        r.validate_model(model)
        model.trainable = False
    with timings.measure("reference_setup"):
        reference = r.normalizers(args.stain_reference)
    with timings.measure("integrity"):
        weights_before = r.weight_hashes(model.weights)
    predictions, scores = infer_rows(model,rows,paths,reference,args.batch_size,device,timings,logger,args.log_every)
    with timings.measure("integrity"):
        if weights_before != r.weight_hashes(model.weights):
            raise ValueError("Frozen model changed during evaluation")
        if r.sha256_file(args.model) != sidecar['model_sha256'] or r.sha256_file(args.metadata) != metadata_sha:
            raise ValueError('Checkpoint files changed during evaluation')
    with timings.measure("metrics"):
        metrics = calculate_binary_metrics([row["ground_truth_code"] for row in rows],scores)
    with timings.measure("output"):
        (args.output / "predictions.csv").write_bytes(csv_bytes(predictions,list(predictions[0])))
        scientific = dict(cohort=args.cohort,samples=len(rows),class_counts=dict(Counter(row['class_name'] for row in rows)),
                          metrics=metrics,model_sha256=sidecar["model_sha256"],manifest_sha256=r.sha256_file(args.manifest),
                          threshold="p_tumour > 0.5; equality benign",description=frozen.LABEL)
        if args.cohort == "VAL2":
            scientific["publication_comparison"] = dict(approximate_native_VAL2_accuracy=0.967,
                                                        accuracy_difference_percentage_points=100*(metrics["accuracy"]-0.967),
                                                        caveat="Different training cohort and binary label task; not an exact reproduction")
        r.write_json(args.output / "metrics.json",scientific)
        r.write_json(args.output / 'confusion_matrix.json',dict(matrix=metrics['confusion_matrix'],
                     orientation='rows=true, columns=predicted',label_order=['benign','tumour']))
        environment = r.metadata(sidecar["split"],"frozen_binary_evaluation",args.batch_size,sidecar.get("precision","float32"))
        environment.update(device=device,device_info=device_info,checkpoint_metadata=sidecar)
        environment.update(evaluator_source_sha256=frozen.source_hashes(),
                           frozen_contract=frozen.scientific_contract(),
                           readiness_sha256=r.sha256_file(args.readiness) if readiness else None,
                           input_sha256=readiness['input_sha256'] if readiness else None,
                           checkpoint_and_weights_unchanged=True)
        r.write_json(args.output / "environment.json",environment)
    performance = dict(cohort=args.cohort,device=device,device_info=device_info,samples=len(rows),batch_size=args.batch_size,
                       timing=timings.report(len(rows),time.perf_counter()-start),
                       scope="Serial phases; zero warmup; total/output exclude final timing JSON and completion marker/log self-reporting tail")
    r.write_json(args.output / "timing_summary.json",performance)
    logger.info("Final metrics: %s; timing: %s",metrics,performance)
    r.write_json(args.output / "run_complete.json",dict(completed_at_utc=r.utc_now(),status="PASS",
                 artifacts={p.name:r.sha256_file(p) for p in args.output.iterdir() if p.is_file() and p.suffix in (".json",".csv")}))
    logger.info("Results written to %s; completed UTC=%s",args.output,r.utc_now())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model","metadata","manifest","extracted","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument('--readiness',type=Path,help='Frozen Stage 2I PASS report; mandatory for VAL2')
    parser.add_argument("--cohort",choices=["VAL1","VAL2"],required=True)
    parser.add_argument("--final-evaluation",action="store_true")
    parser.add_argument("--stain-reference",type=Path,default=r.REFERENCE)
    parser.add_argument("--batch-size",type=int,choices=[1,2,4],default=1)
    parser.add_argument("--device",choices=["cpu","gpu"],default="gpu")
    parser.add_argument("--log-every",type=int,default=1000)
    parser.add_argument("--log-level",choices=["INFO","DEBUG"],default="INFO")
    args = parser.parse_args(argv)
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    try:
        return run(args)
    except Exception:
        r.LOG.exception("Evaluation failed; no completion marker")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
