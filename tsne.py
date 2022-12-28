import os
import numpy as np
import logging
import argparse

import torch
import torch.nn as nn

from models import SupResNet, SSLResNet
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    synthesize_OOD,
    EWM,
)
import data
import matplotlib.pyplot as plt
import sklearn.manifold


def show_tsne(ftrain, ftest, food, labelstrain, labelstest, labelsoos, args, num=10000, draw_sbs=True, cls=0, out_dir="temp.png"):
    # cls = 3

    r = 0.1
    delta = 1
    resample = True

    # print(labelstest)

    # -------sbs----------
    ewm=EWM(1)
    sbs = synthesize_OOD(ewm=ewm, feature=torch.FloatTensor(np.copy(ftest[np.where(labelstest==cls)])), near_region=r, delta=delta, resample=resample, lock_boundary=False)
    # sbs = synthesize_OOD(ewm=ewm, feature=torch.FloatTensor(np.copy(ftrain)), near_region=r, delta=delta, resample=resample, lock_boundary=False)
    sbs = sbs.detach().numpy()

    # ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    # ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    # food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
    # sbs /= np.linalg.norm(sbs, axis=-1, keepdims=True) + 1e-10

    # m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    # ftrain = (ftrain - m) / (s + 1e-10)
    # ftest = (ftest - m) / (s + 1e-10)
    # food = (food - m) / (s + 1e-10)
    # sbs = (sbs - m) / (s + 1e-10)

    # ftest = ftest[:num]
    # food = food[:num]
    # print("ftest:", ftest.shape)
    # print("food: ", food.shape)
    # fall = np.concatenate((ftest, food), axis=0)
    # print("fall: ", fall.shape)

    ftest = ftest[np.where(labelstest==cls)]
    num = ftest.shape[0]
    food = food[:1]
    sbs = sbs[:num]
    print("ftest:", ftest.shape)
    print("food:", food.shape)
    print("sbs:", sbs.shape)

    if draw_sbs:
        fall = np.concatenate((ftest, sbs, food), axis=0)
    else:
        fall = np.concatenate((ftest, food), axis=0)
    print("fall: ", fall.shape)
    
    X_tsne = sklearn.manifold.TSNE(n_components=2, init='random', random_state=87, perplexity=7).fit_transform(fall)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize


    plt.scatter(X_norm[-num:, 0], X_norm[-num:, 1], c='red', s=8, label="OOD")
    if draw_sbs:
        plt.scatter(X_norm[num:2*num, 0], X_norm[num:2*num, 1], c='blue', s=8, label="SBS")
    plt.scatter(X_norm[:num, 0], X_norm[:num, 1], c='green', s=8, label="ID")

    plt.legend()
    plt.savefig(out_dir)
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument(
        "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    )
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--ood", type=str, default="cifar100")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, help="checkpoint path")

    parser.add_argument("--run-name", type=str)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--classes", type=int, default=10)
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

    ckpt_name = f"checkpoint_{args.epoch}.pth.tar"

    if not os.path.exists(f"tsne_result/{args.run_name}"):
        os.mkdir(f"tsne_result/{args.run_name}")



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
    features_test, labels_test = get_features(model.encoder, test_loader)
    # print("In-distribution features shape: ", features_train.shape, features_test.shape)

    # ds = ["cifar10", "cifar100", "svhn", "texture", "blobs"]
    ds = args.ood
    # ds = "svhn"

    _, ood_loader, _ = data.__dict__[ds](
        args.data_dir,
        args.batch_size,
        mode="base",
        normalize=args.normalize,
        norm_layer=norm_layer,
        size=args.size,
    )
    features_ood, labels_ood = get_features(model.encoder, ood_loader)
    # print("Out-of-distribution features shape: ", features_ood.shape)

    for i in range(10):
        show_tsne(
            np.copy(features_train),
            np.copy(features_test),
            np.copy(features_ood),
            np.copy(labels_train),
            np.copy(labels_test),
            np.copy(labels_ood),
            args,
            cls=i,
            out_dir=f"tsne_result/{args.run_name}/{i}.png"
        )
    


if __name__ == "__main__":
    main()
