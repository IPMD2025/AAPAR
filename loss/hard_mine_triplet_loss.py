from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mmd_rbf2(x, y, sigmas=None):
    N, _ = x.shape
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = L = P = 0.0
    XX2 = rx.t() + rx - 2*xx
    YY2 = ry.t() + ry - 2*yy
    XY2 = rx.t() + ry - 2*zz

    if sigmas is None:
        sigma2 = torch.mean((XX2.detach()+YY2.detach()+2*XY2.detach()) / 4)
        sigmas2 = [sigma2/4, sigma2/2, sigma2, sigma2*2, sigma2*4]
        alphas = [1.0 / (2 * sigma2) for sigma2 in sigmas2]
    else:
        alphas = [1.0 / (2 * sigma**2) for sigma in sigmas]

    for alpha in alphas:
        K += torch.exp(- alpha * (XX2.clamp(min=1e-12)))
        L += torch.exp(- alpha * (YY2.clamp(min=1e-12)))
        P += torch.exp(- alpha * (XY2.clamp(min=1e-12)))

    beta = (1./(N*(N)))
    gamma = (2./(N*N))

    return F.relu(beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P))

def mmd_loss(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N, C, H, W = f1.shape
        f1 = f1.view(N, -1)
        N, C, H, W = f2.shape
        f2 = f2.view(N, -1)

    if normalized == True:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return _mmd_rbf2(f1, f2, sigmas=sigmas)


class CFLLoss(nn.Module):
    """ Common Feature Learning Loss
        CFL Loss = MMD + MSE
    """
    def __init__(self,  sigmas = [0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True):#Papaer code: sigmas = [0.001, 0.01, 0.05, 0.1, 0.2, 1, 2]
        super(CFLLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized

    def forward(self,hs, ht, _Fs, _Ft, fs, ft):# (hs, ht), (_Fs, _Ft), (fs, ft) ##RECON: (_fs, _ft), (fs, ft); MMD: hs, ht
        mmd = mmd_loss(hs, ht, sigmas=self.sigmas, normalized=self.normalized)
        mse = F.mse_loss(_Fs,fs) + F.mse_loss(_Ft,ft)
        return mmd, mse

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class FullTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feature, label):
        feature = feature.permute(1, 0, 2).contiguous()  # (31*2, 32, 256)
        label = label.unsqueeze(0).repeat(feature.size(0), 1) # (31*2, 32)

        # feature: [n, m, d], label: [n, m], n=31*2, m=batch_size, d=channel
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)

        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        full_loss_metric_mean[full_loss_num == 0] = 0

        # return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num
        return full_loss_metric_mean.mean()

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist