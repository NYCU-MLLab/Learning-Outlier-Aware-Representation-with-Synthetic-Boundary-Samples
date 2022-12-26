import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import AverageMeter, ProgressMeter, accuracy, synthesize_OOD, synthesize_OOD_Sup


def supervised(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    args=None,
    warmup=False,
    ewm=None,
):
    print(
        " ->->->->->->->->->-> One epoch with supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def ssl(
    model,
    device,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler=None,
    epoch=0,
    args=None,
    warmup=False,
    # sbs=False,
    # resample=False,
    # near_OOD=0.1,
    # normalize_ID=False,
    # grad_head=False,
    ewm=None,
):
    print(
        " ->->->->->->->->->-> One epoch with self-supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    fnm = (epoch>=args.fnm_epoch)
    print("false negative masking!!!")

    for i, data in enumerate(dataloader):
        images, target = data[0], data[1].to(device)
        images = torch.cat([images[0], images[1]], dim=0).to(device)
        bsz = target.shape[0]

        # basic properties of training
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        encoded_feature = model.encoder(images)
        # synthesize boundary samples
        if args.virtual_outlier and not warmup and epoch > args.stdepochs:
            with torch.no_grad():
                if args.training_mode == "SupCon":
                    negative_feature = synthesize_OOD_Sup(
                        ewm,
                        F.normalize(
                            encoded_feature.detach(),
                            dim=-1
                        ) if args.normalize_ID else encoded_feature.detach(),
                        target,
                        args,
                    )
                elif args.training_mode == "SimCLR":
                    negative_feature = synthesize_OOD(
                        ewm,
                        F.normalize(
                            encoded_feature.detach(),
                            dim=-1
                        ) if args.normalize_ID else encoded_feature.detach(),
                        args.near_region,
                        args.delta,
                        args.resample,
                        args.lock_boundary,
                    )
                else:
                    raise ValueError("training mode not supported")
                
                if not args.grad_head:
                    negative_features = model.head(negative_feature)

            if args.grad_head:
                negative_features = model.head(negative_feature)
            # print("negative_features size:", encoded_feature.size())
                    
        else:
            negative_features = None

        features = model.head(encoded_feature)
            
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if args.training_mode == "SupCon":
            loss = criterion(features, target, negative_features=negative_features, fnm=fnm)
        elif args.training_mode == "SimCLR":
            loss = criterion(features, negative_features=negative_features, fnm=fnm)
        else:
            raise ValueError("training mode not supported")

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
