# CSED703 Assignment 2

## Requirement

### Environment
* Pytorch
* NVIDIA CUDA 10.0

### pip packages
* pandas seaborn numpy matplotlib

## Run Training

`cd` into `csed703_assn2`. Run `train.py` to train the classifier from scratch. This will delete `checkpoint/checkpoint.chk`.

It will update loss graph as image periodically, and save intermediate confusion matrix as images where its title shows validation accuracy.

## Run Test

`cd` into `csed703_assn2`. Run `test.py` to train the classifier from scratch. This requires `checkpoint/checkpoint.chk`

It will generate confusion matrix plot in the directory. I froze the checkpoint at accuracy around 0.49.