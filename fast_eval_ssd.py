from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
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
    compute_dis,
)
import data

# local utils for SSD evaluation
def get_scores(ftrain, ftest, food, labelstrain, aclusters, atraining_mode):
    # if aclusters == 1:
    return get_scores_one_cluster(ftrain, ftest, food)
    # else:
    #     if atraining_mode == "SupCE":
    #         print("Using data labels as cluster since model is cross-entropy")
    #         ypred = labelstrain
    #     else:
    #         ypred = get_clusters(ftrain, aclusters)
    #     return get_scores_multi_cluster(ftrain, ftest, food, ypred)


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


def get_eval_results(ftrain, ftest, food, labelstrain, aclusters, atraining_mode):
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

    dtest, dood = get_scores(ftrain, ftest, food, labelstrain, aclusters, atraining_mode)

    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr, np.mean(dtest), np.mean(dood)


def fast_eval(
    model,
    train_loader,
    test_loader,
    # norm_layer,
    OODs,
    adataset="cifar10",
    # aood=["cifar100", "svhn", "texture", "blobs"],
    # aexp_name="temp_eval_ssd",
    atraining_mode="SimCLR", #choices=("SimCLR", "SupCon", "SupCE")
    # aresults_dit,
    # aarch,
    # aclasses=10,
    aclusters=1,
    # adata_dir="./data/data_vvikash/fall20/SSD/datasets/",
    # adata_mode="base",   #choices=("org", "base", "ssl")
    # anormalize=False,
    # abatch_size=256,
    # asize=32,
    # agpu="0",
    # aseed=12345
):

    # create model
    model.eval()

    with torch.no_grad():

        features_train, labels_train = get_features(
            model.encoder, train_loader
        )  # using feature befor MLP-head
        features_test, _ = get_features(model.encoder, test_loader)
        
        fpr95 = []
        auroc = []
        aupr = []
        mood = []

        for ood_name, ood_loader in OODs:

            features_ood, _ = get_features(model.encoder, ood_loader)
            # print("Out-of-distribution features shape: ", features_ood.shape)

            _fpr95, _auroc, _aupr, _mtest, _mood = get_eval_results(
                np.copy(features_train),
                np.copy(features_test),
                np.copy(features_ood),
                np.copy(labels_train),
                aclusters, atraining_mode,
            )

            fpr95.append(_fpr95)
            auroc.append(_auroc)
            aupr.append(_aupr)
            mood.append(_mood)
            print(f"In-data = {adataset}, OOD = {ood_name}, Clusters = {aclusters}, FPR95 = {_fpr95}, AUROC = {_auroc}, AUPR = {_aupr}")
        
        mean_dis, max_dis = compute_dis(features_train)
        print("################## mean and max distance: ", mean_dis, max_dis)

    return fpr95, auroc, aupr, _mtest, mood, mean_dis, max_dis, 
