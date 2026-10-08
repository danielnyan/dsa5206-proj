"""Frozen native binary evaluation contract; never reads dataset images."""
from pathlib import Path
import json
import os

from . import reimplementation as r

LABEL = ('Constrained binary reimplementation trained on released VAL1 and evaluated '
         'on held-out VAL2; not an exact reproduction of Tolkach et al.')
SOURCES = ('modern_pca/evaluate_reimplementation.py', 'modern_pca/stage2i_contract.py',
           'modern_pca/reimplementation.py', 'modern_pca/models.py',
           'modern_pca/evaluate_paper.py', 'modern_pca/cached_training.py',
           'tools/build_validation_manifests.py')


def source_hashes():
    return {name: r.sha256_file(Path(name)) for name in SOURCES}


def scientific_contract():
    return dict(model='NASNetLarge', input_shape=[350,350,3],
                head=['Flatten','Dense(256,relu)','Dense(2,softmax)'],
                parameters=r.PARAMETERS, class_order=['benign','tumour'],
                selection='none; fixed epoch14', threshold=0.5,
                decision='p_tumour > 0.5; equality benign', precision='float32',
                TF32=False, training=False, augmentation=False, TTA=False,
                calibration=False, warmup_samples=0, batch_size=1, device='gpu',
                allocator='default', manifest_sha256=r.VAL2_SHA,
                counts={'benign':24471,'tumour':9632}, samples=34103,
                stain_reference_sha256=r.REFERENCE_SHA,
                preprocessing=dict(r.preprocessing_settings(),resize='Pillow LANCZOS 350x350',
                                   dtype='float32',scale='1/255'),
                metrics=['accuracy','precision','recall','sensitivity','specificity','f1','roc_auc'],
                confusion_matrix='[[TN,FP],[FN,TP]]', probabilities='unrounded float32 losslessly promoted to float64; CSV round-trip precision',
                description=LABEL)


def validate_readiness(args, sidecar):
    """Fail before resolving any image paths if the approved configuration drifts."""
    path = getattr(args, 'readiness', None)
    if path is None:
        raise ValueError('Final VAL2 evaluation requires --readiness')
    report = json.loads(Path(path).read_text())
    if report.get('status') != 'PASS' or report.get('frozen_contract') != scientific_contract():
        raise ValueError('Stage 2I readiness/scientific contract mismatch')
    if report.get('source_sha256') != source_hashes():
        raise ValueError('Frozen evaluator/dependency source changed')
    if report.get('software') != r.capture_environment()['software']:
        raise ValueError('Frozen evaluation software changed')
    if (args.cohort != 'VAL2' or args.batch_size != 1 or args.device != 'gpu'
            or os.environ.get('TF_GPU_ALLOCATOR') not in (None, '', 'bfc')):
        raise ValueError('Frozen cohort/batch/device/allocator mismatch')
    for name in ('model','metadata','manifest','stain_reference'):
        if r.sha256_file(getattr(args,name)) != report['input_sha256'][name]:
            raise ValueError(f'Frozen {name} checksum mismatch')
    if sidecar.get('model_sha256') != report['input_sha256']['model']:
        raise ValueError('Frozen checkpoint mismatch')
    return report
