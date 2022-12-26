import os
import sys
import numpy as np
import math
import time
import shutil, errno
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


#### logging ####
def save_checkpoint(state, is_best, results_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(results_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(results_dir, filename),
            os.path.join(results_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

class EWM(object):
    # exponentially weighted moving mean and covariance
    def __init__(self, alpha=0.5):
        self.mean = None
        self.cov = None
        self.alpha = alpha

        self.max_M_dis = 0
        self.max_M_dis_one_epoch = 0

    def update_mean(self, mean_t):
        if self.mean is None:
            self.mean = mean_t

        self.mean = self.alpha * mean_t + (1 - self.alpha) * self.mean
    
    def update_cov(self, cov_t):
        if self.cov is None:
            self.cov = cov_t

        self.cov = self.alpha * cov_t + (1 - self.alpha) * self.cov

    def update_boundary(self, max_M_dis):
        self.max_M_dis = max(self.max_M_dis, max_M_dis)
        self.max_M_dis_one_epoch = max(self.max_M_dis_one_epoch, max_M_dis)

        return self.max_M_dis

    def update_boundary_by_epoch(self):
        # if the max M distance in this epoch is smaller than "max_M_dis", 
        # then decrease "max_M_dis" to the maximum M dis of this eopch
        self.max_M_dis = min(self.max_M_dis, self.max_M_dis_one_epoch)
        self.max_M_dis_one_epoch = 0

        return self.max_M_dis

#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_features(model, dataloader, max_images=10 ** 10, verbose=False):
    features, labels = [], []
    total = 0
    
    model.eval()
    
    for index, (img, label) in enumerate(dataloader):

        if total > max_images:
            break

        img, label = img.cuda(), label.cuda()

        features += list(model(img).data.cpu().numpy())
        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)

        total += len(img)

    return np.array(features), np.array(labels)


def baseeval(model, device, val_loader, criterion, args, epoch=0):
    """
    Evaluating on validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return top1.avg, top5.avg


def knn(model, device, val_loader, criterion, args, writer, epoch=0):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            # compute output
            output = F.normalize(model(images), dim=-1).data.cpu()
            features.append(output)
            labels.append(target)

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


#### OOD detection ####
def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)

def norm_feats(feat):
    feat /= np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-10
    return feat

def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=False):
    # ntest = np.linalg.norm(_ftest, axis=-1)
    # ndood = np.linalg.norm(_food, axis=-1)
    # m, s = np.mean(_ftrain, axis=0, keepdims=True), np.std(_ftrain, axis=0, keepdims=True)

    # ftrain = (_ftrain - m) / (s + 1e-10)
    # ftest = (_ftest - m) / (s + 1e-10)
    # food = (_food - m) / (s + 1e-10)

    # _dtest = np.max( np.matmul(norm_feats(ftest), norm_feats(ftrain).T), axis=1)# * np.linalg.norm(ftest)
    # _dood = np.max( np.matmul(norm_feats(food), norm_feats(ftrain).T), axis=1)# * np.linalg.norm(food)
    # print("t sim range", f"{np.min(_dtest)}~{np.max(_dtest)}, mean:{np.mean(_dtest)}")
    # print("o sim range", f"{np.min(_dood)}~{np.max(_dood)}, mean:{np.mean(_dood)}")
    # print("t norm range", f"{np.min(ntest)}~{np.max(ntest)}, mean:{np.mean(ntest)}")
    # print("o norm range", f"{np.min(ndood)}~{np.max(ndood)}, mean:{np.mean(ndood)}")

    # _ftrain /= np.linalg.norm(_ftrain, axis=-1, keepdims=True) + 1e-10
    # _ftest /= np.linalg.norm(_ftest, axis=-1, keepdims=True) + 1e-10
    # _food /= np.linalg.norm(_food, axis=-1, keepdims=True) + 1e-10

    # m, s = np.mean(_ftrain, axis=0, keepdims=True), np.std(_ftrain, axis=0, keepdims=True)

    # ftrain = (_ftrain - m) / (s + 1e-10)
    # ftest = (_ftest - m) / (s + 1e-10)
    # food = (_food - m) / (s + 1e-10)

    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
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
    # print("t M range", f"{np.min(dtest)}~{np.max(dtest)}, mean:{np.mean(dtest)}")
    # print("o M range", f"{np.min(dood)}~{np.max(dood)}, mean:{np.mean(dood)}")
    # dtest /=  (ntest/-np.log(_dtest))
    # dood /= (ndood/-np.log(_dood))

    return dtest, dood


def compute_dis(features_train):

    dis = np.linalg.norm(features_train - np.mean(features_train, axis=0, keepdims=True), axis=-1)
    # print("dis size: ", dis.shape)

    return np.mean(dis), np.max(dis)


#### Dataloaders ####
def readloader(dataloader):
    images = []
    labels = []
    for img, label in dataloader:
        images.append(img)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def unnormalize(x, norm_layer):
    m, s = (
        torch.tensor(norm_layer.mean).view(1, 3, 1, 1),
        torch.tensor(norm_layer.std).view(1, 3, 1, 1),
    )
    return x * s + m


class ssdk_dataset(torch.utils.data.Dataset):
    def __init__(self, images, norm_layer, copies=1, s=32):
        self.images = images

        # immitating transformations used at training self-supervised models 
        # replace it if training models with a different data augmentation pipeline
        self.tr = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(s, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                norm_layer,
            ]
        )

        self.n = len(images)
        self.size = len(images) * copies

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.tr(self.images[idx % self.n]), 0


def sliceloader(dataloader, norm_layer, k=1, copies=1, batch_size=128, size=32):

    images, labels = readloader(dataloader)
    indices = np.random.permutation(np.arange(len(images)))
    images, labels = images[indices], labels[indices]

    index_k = torch.cat(
        [torch.where(labels == i)[0][0:k] for i in torch.unique(labels)]
    ).numpy()
    index_not_k = np.setdiff1d(np.arange(len(images)), index_k)

    dataset_k = ssdk_dataset(
        unnormalize(images[index_k], norm_layer), norm_layer, copies, size
    )
    dataset_not_k = torch.utils.data.TensorDataset(
        images[index_not_k], labels[index_not_k]
    )
    print(
        f"Number of selected OOD images (k * num_classes_ood_dataset) = {len(index_k)} \nNumber of OOD images after augmentation  = {len(dataset_k)} \nRemaining number of test images in OOD dataset = {len(dataset_not_k)}"
    )

    loader_k = torch.utils.data.DataLoader(
        dataset_k, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    loader_not_k = torch.utils.data.DataLoader(
        dataset_not_k,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader_k, loader_not_k


def synthesize_OOD(ewm, feature, near_region, delta, resample, lock_boundary):
    device = torch.device("cuda") if feature.is_cuda else torch.device("cpu")

    #------------------ outlier projection-----------------------------
    # compute mean and covariance
    # contrast_feature: [bs*2, feature_dim], mean: [feature_dim], cov: [feature_dim, feature_dim]
    con_f = feature.detach().cpu().numpy()
    #@@@@@@@@@@@@@@@@@@@@@@test
    # norm_f = np.linalg.norm(con_f, axis=-1, keepdims=True)
    # con_f /= norm_f + 1e-10
    #@@@@@@@@@@@@@@@@@@@@@@test
    mu = np.mean(con_f, axis=0, keepdims=True)
    ewm.update_mean(mu)
    mu = ewm.mean

    cov = np.cov(con_f.T, bias=True)
    # TODO compute cov with ewm mean
    ewm.update_cov(cov)
    cov = ewm.cov

    # find maximum Mahalanobis distance
    deviation = con_f - mu
    M_dis = np.sum(
        deviation * (
            np.linalg.pinv(cov).dot(
                deviation.T
            )
        ).T,
        axis=-1,
    )

    # update the boundary distance
    boundary_dis = ewm.update_boundary(M_dis.max())
    if not lock_boundary:
        boundary_dis = M_dis.max()

    # Defined Near OOD region (todo set good nearood)
    nearood_low = boundary_dis * (1 + delta)
    nearood_high = boundary_dis * (1 + delta + near_region)

    # sample z and compute the Mahalanobis distance of z
    if resample:
        sampled_z = np.random.multivariate_normal(np.squeeze(mu), cov, con_f.shape[0])
        deviation = sampled_z - mu
        M_dis_z = np.sum(
            deviation * (
                np.linalg.pinv(cov).dot(
                    deviation.T
                )
            ).T,
            axis=-1,
        )
        # sampled_z = torch.from_numpy(sampled_z).float().to(device)
        # print("M_dis_z: ", M_dis_z.size())

    else:
        M_dis_z = M_dis

    # compute project scaler (c)
    project_scalar = np.sqrt((np.random.uniform(nearood_low, nearood_high, M_dis.shape)) / M_dis_z)

    # project contrast_feature onto near OOD region
    negative_feature = np.expand_dims(project_scalar, 1) * deviation + mu
    #@@@@@@@@@@@@@@@@@@@@@@test
    # negative_feature /= np.linalg.norm(negative_feature, axis=-1, keepdims=True) + 1e-10
    # negative_feature *= norm_f
    #@@@@@@@@@@@@@@@@@@@@@@test
    negative_feature = torch.from_numpy(negative_feature).float().to(device)

    return negative_feature


def synthesize_OOD_Sup(ewm, feature, labels, args): #near_region, delta, resample, lock_boundary):
    device = torch.device("cuda") if feature.is_cuda else torch.device("cpu")
    label = labels.repeat(2).detach().cpu().numpy().astype(int)
    
    #------------------ outlier projection-----------------------------
    # compute mean and covariance
    # contrast_feature: [bs*2, feature_dim], mean: [feature_dim], cov: [feature_dim, feature_dim]
    con_f = feature.detach().cpu().numpy()
    for i in range(args.num_classes):
        index = np.where(label == i)[0]
        if index.shape[0] == 0:
            continue

        class_f = con_f[index]
        mu = np.mean(class_f, axis=0, keepdims=True)
        ewm[i].update_mean(mu)

    cov = np.cov(con_f.T, bias=True)
    ewm[0].update_cov(cov)
    
    mu = np.array([ewm[i].mean for i in label]).reshape(con_f.shape)
    cov = ewm[0].cov

    # find maximum Mahalanobis distance
    deviation = con_f - mu
    M_dis = np.sum(
        deviation * (
            np.linalg.pinv(cov).dot(
                deviation.T
            )
        ).T,
        axis=-1,
    )

    # update the boundary distance
    class_max_M = [0] * args.num_classes
    if args.lock_boundary:
        for i in range(label.shape[0]):
            class_max_M[label[i]] = ewm[label[i]].update_boundary(M_dis[i])
    else:
        for i in range(label.shape[0]):
            class_max_M[label[i]] = max(class_max_M[label[i]], M_dis[i])

    # Defined Near OOD region (todo set good nearood)
    nearood_low = np.array([class_max_M[i] * (1 + args.delta) for i in label])
    nearood_high = np.array([class_max_M[i] * (1 + args.delta + args.near_region) for i in label])

    # sample z and compute the Mahalanobis distance of z
    if args.resample:
        raise ValueError("resample for SupCon not complete yet")
        # sampled_z = np.random.multivariate_normal(np.squeeze(mu), cov, con_f.shape[0])
        # deviation = sampled_z - mu
        # M_dis_z = np.sum(
        #     deviation * (
        #         np.linalg.pinv(cov).dot(
        #             deviation.T
        #         )
        #     ).T,
        #     axis=-1,
        # )
        # # sampled_z = torch.from_numpy(sampled_z).float().to(device)
        # # print("M_dis_z: ", M_dis_z.size())

    else:
        M_dis_z = M_dis

    # compute project scaler (c)
    project_scalar = np.sqrt((np.random.uniform(nearood_low, nearood_high, M_dis.shape)) / M_dis_z)

    # project contrast_feature onto near OOD region
    negative_feature = np.expand_dims(project_scalar, 1) * deviation + mu
    #@@@@@@@@@@@@@@@@@@@@@@test
    # negative_feature /= np.linalg.norm(negative_feature, axis=-1, keepdims=True) + 1e-10
    # negative_feature *= norm_f
    #@@@@@@@@@@@@@@@@@@@@@@test
    negative_feature = torch.from_numpy(negative_feature).float().to(device)

    return negative_feature
