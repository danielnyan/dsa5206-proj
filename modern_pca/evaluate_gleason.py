"""Evaluate frozen models, or mine training-only multipattern patches (>0.95)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .gleason import (dataset,read_rows,verify_cache,load_model,gpu_setup,
                      write_csv,write_json,provenance,CLASSES,sha,
                      select_training,assert_training_isolated,mining_decision,manifest_datasets)


def mining_candidates(manifests, prepared, meta):
    if (prepared['sample_only'] or meta.get('smoke') or not meta.get('complete')
            or meta.get('training_cohort') != 'pure'):
        raise ValueError('Mining requires a complete, non-smoke initial pure-cohort model and full data')
    if meta['manifests'] != prepared['manifests']:
        raise ValueError('Mining manifests differ from the initial model provenance')
    if not prepared.get('region_grade_sha256') or meta.get('region_grade_sha256') != prepared['region_grade_sha256']:
        raise ValueError('Mining requires frozen source primary/secondary grade metadata')
    sources = manifest_datasets(prepared)
    if meta.get('datasets', ['crowd', 'sicap']) != sources:
        raise ValueError('Cross-track mining teacher is forbidden')
    train = sum([read_rows(manifests,d,'train',allow_unlabelled=True) for d in sources],[])
    candidates = [r for r in train if r.get('region_role') == 'mixed'
                  and (r['dataset'] == 'gleason2019' or (str(r.get('region_primary')) in ('3','4','5')
                  and str(r.get('region_secondary')) in ('3','4','5')
                  and str(r['region_primary']) != str(r['region_secondary'])))]
    if not candidates:
        raise ValueError('No documented multipattern training candidates')
    heldout = sum([read_rows(manifests,d,s) for d in sources
                   for s in ('validation','test') if f'{d}_{s}.csv' in prepared['manifests']],[])
    assert_training_isolated(select_training(train,'pure') + candidates, heldout)
    return candidates


def mining_records(rows, probabilities, model_sha):
    import numpy as np
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(rows),3):
        raise ValueError('Prediction count/class shape does not match candidates')
    predictions, retained = [], []
    for row, values in zip(rows, probabilities):
        if row['split'] != 'train' or row.get('region_role') != 'mixed':
            raise ValueError('Only documented multipattern training rows may be mined')
        label = mining_decision(values)
        evidence = dict(p_gp3=float(values[0]),p_gp4=float(values[1]),p_gp5=float(values[2]),
                        mining_model_sha256=model_sha, mining_threshold=0.95,
                        annotation_label=row['label'], annotation_class_name=row['class_name'],
                        annotation_label_source=row['label_source'])
        predictions.append(dict(row, **evidence, selected=label is not None,
                                mined_label=label if label is not None else ''))
        if label is not None:
            retained.append(dict(row, **evidence, label=label, class_name=CLASSES[label],
                                 label_source='initial_pure_model_probability_gt_0.95'))
    return predictions, retained


def mine(args, prepared):
    # Inspect teacher provenance and candidate identities before allocating a GPU.
    meta = json.loads((args.model_dir/'run.json').read_text())
    candidates = mining_candidates(args.manifests, prepared, meta)
    verify_cache(args.cache,candidates)
    gpu_setup(42)
    model,meta = load_model(args.model_dir)
    args.output.mkdir(parents=True,exist_ok=False)
    probabilities = model.predict(dataset(candidates,args.cache,args.batch_size),verbose=2)
    predictions,retained = mining_records(candidates,probabilities,meta['model_sha256'])
    write_csv(args.output/'candidate_predictions.csv',predictions)
    fields = list(candidates[0]) + ['p_gp3','p_gp4','p_gp5','mining_model_sha256',
                                  'mining_threshold','annotation_label','annotation_class_name',
                                  'annotation_label_source']
    write_csv(args.output/'mined_train.csv',retained,fields)
    write_json(args.output/'mining.json',dict(provenance(),complete=True,
               initial_model_sha256=meta['model_sha256'],initial_run_sha256=sha(args.model_dir/'run.json'),
               manifests=prepared['manifests'],region_grade_sha256=prepared['region_grade_sha256'],
               class_order=CLASSES,threshold=0.95,comparison='strictly greater than',
               candidates=len(candidates),retained=len(retained),
               retained_by_class={k:sum(r['class_name']==k for r in retained) for k in CLASSES},
               artifacts={name:sha(args.output/name) for name in ('candidate_predictions.csv','mined_train.csv')},
               subsequent_training='not run; requires an explicitly supported training procedure'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true',help='Validation-only bounded execution; never reads test images')
    mode.add_argument('--final-evaluation',action='store_true')
    mode.add_argument('--mine',action='store_true',help='Frozen initial pure-model inference on multipattern training patches only')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('Positive batch required')
    prepared = json.loads((args.manifests.parent/'prepared.json').read_text())
    sources = manifest_datasets(prepared)
    if prepared['sample_only'] and not args.smoke:
        raise ValueError('Sample manifests cannot support mining or final evaluation')
    if args.mine:
        mine(args, prepared)
        return
    gpu_setup(42)
    model,meta = load_model(args.model_dir,allow_smoke=args.smoke)
    if meta.get('datasets', ['crowd', 'sicap']) != sources:
        raise ValueError('Cross-track evaluation model is forbidden')
    if meta['manifests'] != prepared['manifests']:
        raise ValueError('Evaluation manifests differ from frozen training provenance')
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
    metrics = {}
    evaluation_split = 'validation' if args.smoke else prepared.get('evaluation_split', 'test')
    for source in sources:
        rows = read_rows(args.manifests,source,evaluation_split)
        if args.smoke:
            rows = rows[:32]
        verify_cache(args.cache,rows)
        probabilities = model.predict(dataset(rows,args.cache,args.batch_size),verbose=2)
        if probabilities.shape != (len(rows),3) or not np.isfinite(probabilities).all():
            raise ValueError('Invalid predictions')
        records = [dict(sample_id=r['sample_id'],source_wsi=r['source_wsi'],truth=r['label'],
                        p_gp3=float(v[0]),p_gp4=float(v[1]),p_gp5=float(v[2]),predicted=int(v.argmax()))
                   for r,v in zip(rows,probabilities)]
        write_csv(args.output/f'{source}_predictions.csv',records)
        y = [r['label'] for r in rows]
        pred = probabilities.argmax(axis=1)
        kappa = cohen_kappa_score(y,pred,labels=[0,1,2])
        metrics[source] = dict(count=len(rows),confusion=confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
                               cohen_kappa=float(kappa) if np.isfinite(kappa) else None,
                               report=classification_report(y,pred,labels=[0,1,2],target_names=CLASSES,
                                                            output_dict=True,zero_division=0))
    write_json(args.output/'evaluation.json',dict(provenance(),model_sha256=meta['model_sha256'],
               class_order=CLASSES,metrics=metrics,split=evaluation_split,smoke=args.smoke,complete=True))


if __name__ == '__main__':
    main()
