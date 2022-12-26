"""
https://github.com/HobbitLong/SupContrast
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import synthesize_OOD
import time


def get_fnmask(features, device):
    # feature: [bs*2, feature_dim]
    feature = features.detach().cpu().numpy()

    feature /= np.linalg.norm(feature, axis=-1, keepdims=True) + 1e-10
    m, s = np.mean(feature, axis=0, keepdims=True), np.std(feature, axis=0, keepdims=True)
    feature = (feature - m) / (s + 1e-10)

    # print("feature_dim:", int(feature.shape[0]/2))
    mask = np.eye(int(feature.shape[0]/2))
    mask = np.tile(mask, (2, 2)).astype("int")

    # feature_matrix: [bs*2, bs*2, feature_dim]
    feature_matrix = np.repeat(np.expand_dims(feature, 0), feature.shape[0], axis=0)
    
    # mu: [bs*2, bs*2, feature_dim]
    mu = np.repeat(np.expand_dims(feature, 1), feature.shape[0], axis=1)
    dev = feature_matrix - mu

    # cov: [feature_dim, feature_dim]
    cov = np.cov(feature.T, bias=True)

    M_dis = np.sum(
        dev * np.transpose(
            np.matmul(
                np.linalg.pinv(cov),
                np.transpose(dev, (0, 2, 1))
            ),  
            (0, 2, 1)
        ),
        axis=-1,
    )

    M_dis[mask] = 1e10

    ind_x = np.argmax(M_dis, axis=-1, keepdims=False)
    ind_y = np.arange(ind_x.shape[0])

    fnmask = np.ones((feature.shape[0], feature.shape[0]))
    fnmask[ind_y, ind_x] = 0

    return torch.tensor(fnmask).to(device)


class VOConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, vos_mode="Cont", base_temperature=0.07, lamb = 1):
        super(VOConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.vos_mode = vos_mode
        self.lamb = lamb
        # self.resample = resample
        # self.near_OOD = near_OOD

    def forward(self, features, labels=None, negative_features=None, fnm=False):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        features = F.normalize(features, dim=-1)

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is None:
            class_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            class_mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("unbind: ", torch.unbind(features, dim=1))
        # print("fs: ", features.size())
        # print("con size: ", contrast_feature.size())
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # tile mask
        class_mask = class_mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(class_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )

        mask = class_mask * logits_mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        fntime = time.time()
        print("start getting mask")
        # remove false negative
        if fnm:
            fnmask = get_fnmask(contrast_feature, device)
            logits_mask *= fnmask
        print("got mask:", time.time() - fntime)
        
        log_prob = (mask * logits).sum(1) / mask.sum(1)
        exp_logits = (torch.exp(logits) * logits_mask).sum(1)

        if negative_features is not None:
            negative_features = F.normalize(negative_features, dim=-1)
            
            # compute negative logits
            negative_dot_contrast = torch.div(
                torch.matmul(anchor_feature, negative_features.T), self.temperature
            )

            # for numerical stability
            # n_logits_max, _ = torch.max(negative_dot_contrast, dim=1, keepdim=True)
            n_logits = negative_dot_contrast - logits_max.detach()
            
            # if self.vos_mode == "ContOne":
            #     n_exp_logits = (torch.exp(n_logits) * (1 - logits_mask)).sum(1)
            # else:
            if labels is None:
                n_exp_logits = torch.exp(n_logits).sum(1)
            else:
                n_exp_logits = (torch.exp(n_logits) * class_mask).sum(1)
            #------------------------------------------------------------------

            # compute log_prob = - L_cont
            # exp_logits = (torch.exp(logits) * logits_mask).sum(1)
        
            if self.vos_mode == "Cont" or self.vos_mode == "ContOne":
                log_prob -= torch.log(exp_logits + self.lamb * n_exp_logits)
            elif self.vos_mode == "DualCont":
                log_prob += torch.log(1 / exp_logits + self.lamb / n_exp_logits) #dual
            elif self.vos_mode == "DualOut":
                log_prob = (1 + self.lamb) * log_prob - (torch.log(exp_logits) + self.lamb * torch.log(n_exp_logits)) #dual_out
            else:
                raise ValueError("Unknown vos_mode: {}".format(self.vos_mode))
        # print("\n\nafter:", log_prob.size())
        else:
            log_prob -= torch.log(exp_logits)

        # loss
        loss = -(self.temperature / self.base_temperature) * log_prob
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
