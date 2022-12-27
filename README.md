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

---

## Extend work

# False Negative Masking for Contrastive Learning

## Environment Setup
- Install the required packages
```
pip install -r requirements.txt
```

## Run Training

```
python cls_train.py --dataset cifar10 --num-classes 10 --results-dir path --exp-name name --warmup --normalize --fnm-epoch 350
```
- `--result_dir`: path folder for storing the evaluation result
- `--fnm-epoch`: how many epochs of training before starting to use the false negative masking


## Linear Evaluation

The model is evaluated by training a linear classifier after fixing the learned embedding.


```
python ./cls_linear.py --ckpt ckpt_path --result_dir path --dataset cifar10 --classes 10
```
- `--ckpt`: path for the model checkpoint
- `--result_dir`: path folder for storing the evaluation result

