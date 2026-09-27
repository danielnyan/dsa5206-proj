# DSA5206 Project Folder
This GitHub project is for DSA5206 - Advanced Topics in Data Science. The code was adapted from [High-accuracy prostate cancer pathology using deep learning by Tolkach Y. et al.](https://github.com/gagarin37/deep_learning_pca). Please refer to the original repository for more information regarding how to train, finetune and reuse the model. 

# Installation

Begin by checking out COLAB_RUNBOOK.md. The notebook runs on Colab, although it times out when you try to run full training. Might need to convert this into a Python script that could be run locally. Alternatively, you could run it as a Jupyter notebook on your laptop. 

Do note that full reproduction is not possible as Tolkach Y. used manually labelled TCGA data. Instead, this study splits the "validation data" published on Zenodo and uses that as the training / validation dataset instead. 