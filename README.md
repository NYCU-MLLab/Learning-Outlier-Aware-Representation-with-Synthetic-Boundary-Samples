# Learning Outlier-Aware Representation with Synthetic Boundary Samples

This is the PyTorch implementation for Learning Outlier-Aware Representation with Synthetic Boundary Samples from National Yang Ming Chiao Tung University, Taiwan.

## Environment Setup
- Install the required packages
```
pip install -r requirements.txt
```

## Run Training

```
CUDA_VISIBLE_DEVICES=0 python -u train.py --arch resnet18 --training-mode SimCLR --dataset cifar100 --num-classes 100 --results-dir path --exp-name name --warmup --normalize --virtual-outlier --lamb 1 --near-region 0.01 --default-warmup --alpha 0.1 --lock-boundary
```
- arguments:
   - `--arch`: model architecture
   - `--training-mode`: SimCLR, SupCon
   - `--virtual-outlier`: using synthetic boundary samples during training

## Run Testing

```
python ./draw_his.py --training-mode SimCLR --normalize --ckpt ./compare_ckp/cifar100 --dataset cifar100 --classes 100
```
- `--ckpt`: the folder contains the checkpoint name `checkpoint_500.pth.tar`

## Run Tsne

```
python -u tsne.py --training-mode SimCLR --arch resnet18 --dataset cifar10 --normalize --run-name name --ckpt path_to_checkpoint
```
