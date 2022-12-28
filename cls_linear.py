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

import models as md
import data
from utils import knn, accuracy, AverageMeter, ProgressMeter

model_dict = {
    "resnet18": [md.resnet18, 512],
    "resnet34": [md.resnet34, 512],
    "resnet50": [md.resnet50, 2048],
    "resnet101": [md.resnet101, 2048],
}

class Net(nn.Module):
    def __init__(self, num_class, arch="resnet18"):
        super(Net, self).__init__()
        m, fdim = model_dict[arch]
        self.encoder = m()
        self.fc = nn.Linear(fdim, num_class, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument(
        "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    )
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10")
    # parser.add_argument("--OOD", type=str, default="cifar100")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--ckpt", type=str, help="checkpoint path", default="/home/thesis/cls_test/cls_test/6--dataset-cifar10-arch-resnet18-lr-0.5_epochs-500/checkpoint/checkpoint_500.pth.tar")
    parser.add_argument("--result_dir", type=str, default="./linear")

    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument(
        "--data-dir", type=str, default="./datasets/"
    )
    parser.add_argument(
        "--data-mode", type=str, choices=("org", "base", "ssl"), default="org"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--size", type=int, default=32)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    device = "cuda:0"

    assert args.ckpt, "Must provide a checkpint for evaluation"

    # ckpt_name = "checkpoint_500.pth.tar"
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)


    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("my-logger")
    logger.addHandler(
        logging.FileHandler(
            os.path.join(args.result_dir, "result.txt"), "w")
        )
    logger.propagate = False
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model
    
    model = Net(args.classes, args.arch)
    model.encoder = nn.DataParallel(model.encoder).to(device)
    # model = model.to(device)
    # print(model)

    # load checkpoint
    temp = md.SSLResNet(args.arch)
    temp.encoder = nn.DataParallel(temp.encoder).to(device)

    ckpt_dict = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    temp.load_state_dict(ckpt_dict)
    # print(temp)

    model.encoder = temp.encoder
    temp = None
    # model.encoder = nn.DataParallel(model.encoder).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.fc = nn.DataParallel(model.fc)
    print(model)
    # exit()

    # dataloaders
    train_loader, test_loader, _ = data.__dict__[args.dataset](
        args.data_dir,
        mode="org",
        normalize=args.normalize,
        batch_size=args.batch_size,
    )

    # optimizer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    # criterion
    criterion = nn.CrossEntropyLoss()

    # prec1, _ = knn(model, device, test_loader, criterion, args, 0)
    # print("knn:", prec1)
    # knn(model, device, val_loader, criterion, args, writer, epoch=0)
    # print(
    #     f"In-data = {args.dataset}, OOD = {d}, Clusters = {args.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
    # )

    # logger.info(
    #     f"In-data = {args.dataset}, OOD = {d}, Clusters = {args.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
    # )
    
    for epoch in range(args.epochs):
        model.train()
        end = time.time()

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc_1", ":6.2f")
        top5 = AverageMeter("Acc_5", ":6.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )
        best_top1 = 0
        best_top5 = 0

        for i, _data in enumerate(train_loader):
            images, target = _data[0].to(device), _data[1].to(device)

            # basic properties of training
            if i == 0:
                print(
                    "images :",
                    images.shape,
                    "target :",
                    target.shape,
                    f"Batch_size from args: {args.batch_size}",
                    "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
                )
                print(
                    "Pixel range for training images : [min: {}, max: {}]".format(
                        torch.min(images).data.cpu().numpy(),
                        torch.max(images).data.cpu().numpy(),
                    )
                )

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            best_top1 = max(best_top1, acc1[0].data.cpu().numpy())
            best_top5 = max(best_top1, acc5[0].data.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

    print("-------------------------------results----------------------------")
    print(f"Top1 acc: {acc1[0].data.cpu().numpy()} (best:{best_top1})\nTop5 acc: {acc5[0].data.cpu().numpy()} (best:{best_top5})")
    logger.info(
        f"Top1 acc: {acc1[0].data.cpu().numpy()} (best:{best_top1})\nTop5 acc: {acc5[0].data.cpu().numpy()} (best:{best_top5})"
    )


if __name__ == "__main__":
    main()
