# Local smoke-test results

Date: 2026-09-27  
Runtime: TensorFlow 2.21.0, Keras 3.15.1, CPU-only 16 GB Azure VM

The tests used the 30 real histology images shipped by the authors. Metrics from
randomly initialized models are deliberately not interpreted as model quality.

## Exact paper architecture

- NASNetLarge at 350 x 350 pixels
- Original `Flatten -> Dense(256) -> Dense(3)` head
- 209,813,077 parameters
- One real batch-1 forward pass: 0.740 images/s
- One real batch-1 backward/update pass: 0.0346 images/s
- Saved `.keras` model: 843,035,491 bytes
- Save/reload prediction equality: passed (`atol=1e-6`)
- Native C1 and eight-transform C8 prediction shapes: `(1, 3)`
- Complete three-image held-out C1/C8 code path: passed

## Complete training CLI

To keep the local check small, the full epoch/checkpoint/evaluation path used
EfficientNetV2-B0 with the GAP alternative head and random initialization:

- 24 training, 3 validation, and 3 test images
- one full epoch (12 batches), validation, save, reloadable final model, CSV and
  JSON summary: passed
- elapsed time: 48.25 seconds on CPU

GPU execution remains a Colab responsibility because this VM has no CUDA device.
The notebook runs CPU and GPU benchmarks in separate processes to make placement
explicit.

The NumPy Macenko replacement was also executed on an author tumour patch using
the bundled reference stain image; it returned a valid 350 x 350 RGB uint8 image.
