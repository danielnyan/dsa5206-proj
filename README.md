# DSA5206 Project Folder
This GitHub project is for DSA5206 - Advanced Topics in Data Science. The code was adapted from [MedMamba by YubiaoYue](https://github.com/YubiaoYue/MedMamba). Please refer to the original repository for more information regarding how to train, finetune and reuse the model. 

# Installation

For model construction, checkpoint inspection, and the portable CPU smoke test:

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```

For GPU training, first install a CUDA-enabled PyTorch build appropriate for
your driver, then install the optimized scan extension:

```bash
python -m pip install --no-build-isolation -r requirements-gpu.txt
```