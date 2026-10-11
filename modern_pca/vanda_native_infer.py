"""Research-only inference from the verified Vanda 14-epoch binary NASNetLarge checkpoint.

Run from project root: python -m modern_pca.vanda_native_infer --model ... --image ... --output ...
Requires the repository's original stain reference and preprocessing dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA = '433dd48aae800c659d1cba092af2859499bb804d2228ff978a6e9c9602b2adde'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.model.is_file() or not args.image.is_file():
        raise FileNotFoundError('Model or input image is missing')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if sha256(args.model) != EXPECTED_SHA:
        raise RuntimeError('Unexpected model SHA256; refusing inference')

    import numpy as np
    import tensorflow as tf
    from . import reimplementation as r

    tf.keras.mixed_precision.set_global_policy('float32')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = tf.keras.models.load_model(args.model, compile=False)
    r.validate_model(model)
    model.trainable = False
    reference = r.normalizers(r.REFERENCE)
    row = {'sample_id': args.image.name, 'file_sha256': sha256(args.image)}
    x = r.preprocess_legacy(row, args.image, reference)
    probabilities = np.asarray(model(np.expand_dims(x, 0), training=False).numpy()[0], dtype=np.float64)
    if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise RuntimeError('Invalid prediction')
    if not np.all((probabilities >= -1e-6) & (probabilities <= 1 + 1e-6)) or abs(probabilities.sum() - 1) > 1e-5:
        raise RuntimeError('Invalid two-class softmax output')
    p_tumour = float(probabilities[1])
    result = {
        'status': 'PASS',
        'purpose': 'research-only_binary_patch_inference',
        'model_sha256': EXPECTED_SHA,
        'image_sha256': row['file_sha256'],
        'image': str(args.image),
        'p_benign': float(probabilities[0]),
        'p_tumour': p_tumour,
        'tumour_threshold': 0.5,
        'predicted_class': 'tumour' if p_tumour > 0.5 else 'benign',
        'preprocessing': 'legacy_Lanczos350_brightness_Macenko_float32_div255',
        'clinical_use': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
