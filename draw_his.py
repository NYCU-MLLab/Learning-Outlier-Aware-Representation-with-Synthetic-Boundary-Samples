from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

import torch
import torch.nn as nn

from models import SupResNet, SSLResNet
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import data
import matplotlib.pyplot as plt
import seaborn as sns

# local utils for SSD evaluation
def get_scores(ftrain, ftest, food):
    cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood


def draw_histogram(dtest, dood, args, oodset):
    # freq, bins = np.histogram(dtest, bins=100)
    bins_num = 350 #args.hist_range // 200
    # ---histogram---
    plt.clf()
    plt.hist(dtest, alpha=0.5, bins=bins_num, range=[0, args.hist_range], color="g", label="ID")
    plt.hist(dood, alpha=0.5, bins=bins_num, range=[0, args.hist_range], color="r", label="OOD")
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency', xlabel="M Distance")
    plt.xlim(-10, args.hist_range)
    plt.ylim(-10, 500)
    plt.legend()
    # plt.show()
    plt.savefig(f"{args.ckpt}/vs_{oodset}.png")

    # ---density---
    # if oodset == "blobs":
    #     plt.clf()
    #     sns.set_style("white")
    #     kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':2})

    #     plt.figure()#(figsize=(10,7), dpi= 80)
    #     # sns.distplot(dtest, bins=int(2000), kde=True, color="green", label="ID", **kwargs)
    #     # sns.distplot(dood, bins=int(2000), kde=True, color="red", label="OOD", **kwargs)
    #     sns.kdeplot(dtest, shade=True, color="red", label="ID")
    #     sns.kdeplot(dood, shade=True, color="green", label="OOD")
    #     # plt.xlim(-10,7000)
    #     # plt.ylim(-10, 500)
    #     # plt.legend()
    #     plt.show()

def get_eval_results(ftrain, ftest, food, labelstrain, args, oodset):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food)

    draw_histogram(dtest, dood, args, oodset)
    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr


def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument(
        "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    )
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    # parser.add_argument("--OOD", type=str, default="cifar100")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")
    parser.add_argument("--hist-range", type=int, default=7000)

    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=1)
    parser.add_argument(
        "--data-dir", type=str, default="./datasets/"
    )
    parser.add_argument(
        "--data-mode", type=str, choices=("org", "base", "ssl"), default="base"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--size", type=int, default=32)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    device = "cuda:0"

    assert args.ckpt, "Must provide a checkpint for evaluation"

    ckpt_name = "checkpoint_500.pth.tar"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("my-logger")
    logger.addHandler(
        logging.FileHandler(
            os.path.join(args.ckpt, "result.txt"), "w")
        )
    logger.propagate = False
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    if args.training_mode in ["SimCLR", "SupCon"]:
        model = SSLResNet(arch=args.arch).eval()
    elif args.training_mode == "SupCE":
        model = SupResNet(arch=args.arch, num_classes=args.classes).eval()
    else:
        raise ValueError("Provide model class")
    model.encoder = nn.DataParallel(model.encoder).to(device)
    # model = model.to(device)

    # load checkpoint
    ckpt_dict = torch.load(f"{args.ckpt}/{ckpt_name}", map_location="cpu")
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    model.load_state_dict(ckpt_dict)

    # dataloaders
    train_loader, test_loader, norm_layer = data.__dict__[args.dataset](
        args.data_dir,
        args.batch_size,
        mode=args.data_mode,
        normalize=args.normalize,
        size=args.size,
    )

    features_train, labels_train = get_features(
        model.encoder, train_loader
    )  # using feature befor MLP-head
    features_test, _ = get_features(model.encoder, test_loader)
    print("In-distribution features shape: ", features_train.shape, features_test.shape)

    ds = ["cifar10", "cifar100", "svhn", "texture", "blobs"]
    ds.remove(args.dataset)

    for d in ds:
        _, ood_loader, _ = data.__dict__[d](
            args.data_dir,
            args.batch_size,
            mode="base",
            normalize=args.normalize,
            norm_layer=norm_layer,
            size=args.size,
        )
        features_ood, _ = get_features(model.encoder, ood_loader)
        print("Out-of-distribution features shape: ", features_ood.shape)

        fpr95, auroc, aupr = get_eval_results(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            args,
            d,
        )

        print(
            f"In-data = {args.dataset}, OOD = {d}, Clusters = {args.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
        )

        logger.info(
            f"In-data = {args.dataset}, OOD = {d}, Clusters = {args.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
        )


if __name__ == "__main__":
    main()
