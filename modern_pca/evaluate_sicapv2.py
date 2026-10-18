"""Explicit, one-shot SECONDARY epoch15 VAL2 evaluation; never invoked by training."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

from . import reimplementation as r, train_sicapv2 as t
from . import evaluate_reimplementation as e
from .evaluate_paper import calculate_binary_metrics, csv_bytes, Timings


def read(path):
    return json.loads(Path(path).read_text())


def verified_run(root,required):
    complete=read(root/'run_complete.json')
    if complete.get('status')!='PASS' or not set(required).issubset(complete.get('artifacts',{})):
        raise ValueError('Run lacks authenticated completion artifacts')
    for name in required:
        r.sha256_file(root/name,complete['artifacts'][name])


def compare(predictions,baseline):
    if len(predictions)!=len(baseline): raise ValueError('Baseline sample coverage mismatch')
    shifts=[]
    for current,previous in zip(predictions,baseline):
        if (current['sample_id']!=previous['sample_id'] or current['row_index']!=int(previous['row_index'])
                or current['ground_truth_code']!=int(previous['ground_truth_code'])):
            raise ValueError('Baseline sample order/labels mismatch')
        old=float(previous['p_tumour']); new=float(current['p_tumour'])
        if not 0<=old<=1 or int(previous['predicted_binary_code'])!=int(old>0.5):
            raise ValueError('Invalid baseline probability/decision')
        shifts.append(dict(sample_id=current['sample_id'],ground_truth_code=current['ground_truth_code'],
            epoch14_p_tumour=old,epoch15_p_tumour=new,delta_p_tumour=new-old,
            threshold_crossing=int((old>0.5)!=(new>0.5))))
    return shifts


def run(args):
    if not args.final_evaluation: raise ValueError('Explicit --final-evaluation required')
    if os.environ.get('TF_GPU_ALLOCATOR') not in (None,'','bfc'):
        raise ValueError('Match primary baseline default inference allocator')
    plan=t.read_plan(args.plan)
    plan_sha=r.sha256_file(args.plan)
    verified_run(args.run,('metadata.json','epoch15.keras','history.json'))
    meta=read(args.run/'metadata.json')
    if (meta.get('purpose')!=t.PURPOSE or meta.get('completed_epochs')!=15 or meta.get('plan_sha256')!=plan_sha
            or meta.get('protocol')!=t.protocol() or meta.get('training_samples')!=plan['samples']):
        raise ValueError('Require completed, fixed secondary epoch15')
    r.sha256_file(args.run/'epoch15.keras',meta['model_sha256'])
    verified_run(args.baseline,('metrics.json','predictions.csv','environment.json'))
    baseline_metrics=read(args.baseline/'metrics.json')
    baseline_environment=read(args.baseline/'environment.json')
    from .stage2i_contract import scientific_contract, source_hashes
    if (baseline_metrics.get('model_sha256')!=t.MODEL_SHA or baseline_metrics.get('manifest_sha256')!=r.VAL2_SHA
            or baseline_metrics.get('samples')!=34103 or baseline_metrics.get('threshold')!='p_tumour > 0.5; equality benign'
            or baseline_environment.get('frozen_contract')!=scientific_contract()
            or baseline_environment.get('evaluator_source_sha256')!=source_hashes()):
        raise ValueError('Require authenticated Stage 2I native baseline')
    rows=r.read_manifest(args.manifest,'VAL2')
    with (args.baseline/'predictions.csv').open() as stream: baseline=list(csv.DictReader(stream))
    # Check baseline identity before opening images (dummy current probabilities are never used as results).
    compare([dict(row_index=v['row_index'],sample_id=v['sample_id'],ground_truth_code=v['ground_truth_code'],p_tumour=0.0) for v in rows],baseline)
    r.ensure_output_separation(args.extracted,[args.output])
    logger=r.logging_setup(args.output)
    started=time.perf_counter(); timing=Timings()
    import tensorflow as tf
    device=r.gpu_setup()
    model=tf.keras.models.load_model(args.run/'epoch15.keras',compile=False)
    r.validate_model(model); model.trainable=False
    hashes=r.weight_hashes(model.weights)
    checkpoint=read(args.run/'epoch_15_resume/checkpoint.json')
    if hashes!=checkpoint['model_tensor_sha256']:
        raise ValueError('Export differs from completed secondary checkpoint')
    reference=r.normalizers(r.REFERENCE)
    paths=[r.image_path(row,args.extracted,final_evaluation=True) for row in rows]
    predictions,scores=e.infer_rows(model,rows,paths,reference,1,'/GPU:0',timing,logger,1000)
    if hashes!=r.weight_hashes(model.weights): raise ValueError('Evaluation changed model state')
    r.sha256_file(args.run/'epoch15.keras',meta['model_sha256'])
    metrics=calculate_binary_metrics([v['ground_truth_code'] for v in rows],scores)
    old_metrics=calculate_binary_metrics([v['ground_truth_code'] for v in rows],[float(v['p_tumour']) for v in baseline])
    if old_metrics!=baseline_metrics['metrics']: raise ValueError('Baseline metrics do not match predictions')
    shifts=compare(predictions,baseline)
    (args.output/'predictions.csv').write_bytes(csv_bytes(predictions,list(predictions[0])))
    (args.output/'probability_shifts.csv').write_bytes(csv_bytes(shifts,list(shifts[0])))
    crossings=[v for v in shifts if v['threshold_crossing']]
    (args.output/'threshold_crossings.csv').write_bytes(csv_bytes(crossings,list(shifts[0])))
    r.write_json(args.output/'metrics.json',dict(purpose=t.PURPOSE,metrics=metrics,samples=len(rows),
        model_sha256=meta['model_sha256'],manifest_sha256=r.VAL2_SHA,threshold='p_tumour > 0.5; equality benign'))
    keys=('accuracy','precision','recall','sensitivity','specificity','f1','roc_auc')
    r.write_json(args.output/'comparison.json',dict(epoch14=old_metrics,epoch15=metrics,
        differences={key:metrics[key]-old_metrics[key] if metrics[key] is not None and old_metrics[key] is not None else None for key in keys},threshold_crossings=len(crossings),
        interpretation='Secondary external transfer experiment; not paper post-VAL1 reproduction. Report worse results without retrying.',
        descriptive_paper_values=plan['comparison']['descriptive_paper_values']))
    r.write_json(args.output/'environment.json',dict(plan_sha256=plan_sha,device=device,
        software=r.capture_environment()['software'],source_sha256=t.source_hashes(),weights_unchanged=True))
    r.write_json(args.output/'timing_summary.json',dict(seconds=time.perf_counter()-started,timing=timing.report(len(rows),time.perf_counter()-started)))
    r.write_json(args.output/'run_complete.json',dict(status='PASS',artifacts={p.name:r.sha256_file(p) for p in args.output.iterdir() if p.is_file() and p.name!='run.log'}))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','run','baseline','manifest','extracted','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--final-evaluation',action='store_true')
    return run(parser.parse_args(argv))


if __name__=='__main__': raise SystemExit(main())
