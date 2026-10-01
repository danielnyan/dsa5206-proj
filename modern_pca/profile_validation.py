"""Bounded Stage 2E online/cached validation comparison on training samples only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from . import preprocessing_cache as c
from . import reimplementation as r
from . import numerical_parity as p
from .profile_reimplementation import configure, load_model, Utilization, check_originals, compiled_components


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extracted',type=Path,required=True)
    args=parser.parse_args(argv)
    import tensorflow as tf
    root=args.output; r.ensure_output_separation(args.extracted,[root])
    r.logging_setup(root/'validation_comparison')
    info=configure('gpu',asynchronous=True)
    reader=c.CacheReader(root/'cache'); rows=json.loads((root/'samples.json').read_text())[:100]
    start=time.perf_counter(); model,_=load_model(root,r.SCHEDULE[0][0]); model_load_seconds=time.perf_counter()-start
    reference=r.normalizers(r.REFERENCE)
    predict=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec((None,350,350,3),tf.float32)])
    warm=np.stack([c.to_float(reader.get(i)) for i in range(4)])
    start=time.perf_counter(); predict(warm).numpy(); compiled_warmup_seconds=time.perf_counter()-start
    model(warm,training=False).numpy()
    results={}; baseline=None
    for mode in ('cached_eager','cached_compiled','online_compiled'):
        repeats=[]
        for repeat in range(3):
            probabilities=[]
            with Utilization() as monitor:
                start=time.perf_counter()
                if mode.startswith('cached'):
                    batches=(x for x,y,i in c.dataset(reader,list(range(100)),parallel=4,prefetch=2,augment=False))
                else:
                    batches=(np.stack([r.preprocess_legacy(row,r.image_path(row,args.extracted),reference)
                                       for row in rows[offset:offset+4]]) for offset in range(0,100,4))
                for batch in batches:
                    output=model(batch,training=False) if mode=='cached_eager' else predict(batch)
                    probabilities.append(output.numpy())
                seconds=time.perf_counter()-start
            actual=np.concatenate(probabilities)
            if baseline is None:
                baseline=actual
            equivalent=bool(np.allclose(actual,baseline,atol=p.TOLERANCES['forward_probability_atol'],
                                       rtol=p.TOLERANCES['forward_probability_rtol']))
            agreement=bool(np.array_equal(actual[:,1]>.5,baseline[:,1]>.5))
            if not equivalent or not agreement:
                raise ValueError('Validation optimization changed predictions beyond predeclared tolerance')
            repeats.append(dict(seconds=seconds,images=100,images_per_second=100/seconds,
                                max_absolute_difference=float(np.max(np.abs(actual-baseline))),
                                predicted_classes_equal=agreement,parity_pass=equivalent,utilization=monitor.report()))
            r.LOG.info('%s repeat=%d/3 %.2f images/s',mode,repeat+1,100/seconds)
        results[mode]=dict(repeats=repeats,images_per_second=300/sum(x['seconds'] for x in repeats))
    # Match the deepest-stage compiled diagnostic without adding its allocations
    # to the validation measurements above.
    accumulator=r.Accumulator(model,r.adam(1e-5))
    components=compiled_components(model,accumulator,warm,np.array([x['ground_truth_code'] for x in rows[:4]],np.int64))
    r.write_json(root/'first_stage_compiled_components.json',components)
    check_originals(root)
    r.write_json(root/'validation_comparison.json',dict(device=info,results=results,
        model_load_seconds=model_load_seconds,compiled_warmup_seconds=compiled_warmup_seconds,
        scope='100 fixed VAL1 training images, three repeats each; no held-out or VAL2 images; no updates',
        initial_checkpoint_sha256=r.sha256_file(root/'initial.weights.h5'),
        implementation_sha256=r.sha256_file(Path(__file__)),timestamp=r.utc_now()))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
