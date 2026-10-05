"""Evaluate frozen three-output models on the supplied tumour-only test splits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .gleason import (dataset,read_rows,verify_cache,load_model,gpu_setup,
                      write_csv,write_json,provenance,CLASSES)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--manifests',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--smoke',action='store_true',help='Validation-only bounded execution; never reads test images')
    p.add_argument('--final-evaluation',action='store_true')
    args = p.parse_args()
    if args.batch_size < 1 or args.smoke == args.final_evaluation:
        p.error('Choose exactly one of --smoke and --final-evaluation; positive batch required')
    prepared = json.loads((args.manifests.parent/'prepared.json').read_text())
    if prepared['sample_only'] and not args.smoke:
        raise ValueError('Sample manifests cannot support final evaluation')
    gpu_setup(42)
    model,meta = load_model(args.model_dir,allow_smoke=args.smoke)
    if meta['manifests'] != prepared['manifests']:
        raise ValueError('Evaluation manifests differ from frozen training provenance')
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
    metrics = {}
    for source in ('crowd','sicap'):
        rows = read_rows(args.manifests,source,'validation' if args.smoke else 'test')
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
               class_order=CLASSES,metrics=metrics,smoke=args.smoke,complete=True))


if __name__ == '__main__':
    main()
