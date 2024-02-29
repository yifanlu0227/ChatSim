import torch
import torch.nn as nn
import numpy as np
eps = 1e-4

class GammaL1Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = args['weight']
        self.gamma = args['gamma']
        self.alpha = args['alpha']
    
    def forward(self, pred, target, mask=None):
        loss_map = self.alpha * torch.abs(pred**(1/self.gamma) - target**(1/self.gamma))

        if mask is not None:
            loss = (loss_map * mask).sum() / mask.sum() 
        else:
            loss = loss_map.mean()

        return loss * self.weight


class LogEncodedL2Loss(nn.Module):
    def __init__(self, args):
        super(LogEncodedL2Loss, self).__init__()
        self.weight = args['weight']

    def forward(self, pred, target, mask=None):
        loss_map = torch.pow((torch.log(pred+1+eps) - torch.log(target+1+eps)), 2) # N, C, H, W

        if mask is not None:
            loss = (loss_map * mask).sum() / mask.sum() 
        else:
            loss = loss_map.mean()

        return loss * self.weight

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.weight = args['weight']

    def forward(self, pred, target, mask=None):
        loss_map = torch.pow((pred - target), 2) # N, C, H, W

        if mask is not None:
            loss = (loss_map * mask).sum() / mask.sum()
        else:
            loss = loss_map.mean()

        return loss * self.weight
    
class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.weight = args['weight']

    def forward(self, pred, target, mask=None):
        loss_map = torch.abs(pred - target) # N, C, H, W

        if mask is not None:
            loss = (loss_map * mask).sum() / mask.sum()
        else:
            loss = loss_map.mean()

        return loss * self.weight
    
class L1AngularLoss(nn.Module):
    def __init__(self, args):
        super(L1AngularLoss, self).__init__()
        self.weight = args['weight']

    def forward(self, pred, target):
        '''
            pred/target: [B, 3]
        '''
        cosine_similarity = (pred * target).sum(dim=1).clamp(-1+eps, 1-eps)
        angle_diff = torch.acos(cosine_similarity)  # in radians
        loss = torch.abs(angle_diff).mean()

        return loss * self.weight

class AngleClassificationLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = args["weight"]
        self.num_bins = args["num_bins"]
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        pred : torch.tensor
            unnormalized logits, shape [B, num_bins]
        target : torch.tensor
            shape [B, ], ground-truth angle in [0, 2pi]
        """

        bin_indices = (target / (2 * np.pi / self.num_bins)).floor().long()
        loss = self.cross_entropy(pred, bin_indices) * self.weight

        return loss