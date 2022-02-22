""" Here Custom Loss Function like F1/DiceLoss or the AUC calculation are defined
@author: Philipp
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    ###
    # Dice Loss implementation inspired from Kaggle Definition but by now quite adapted
    # https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    # General Formula = 2TP/2 TP +FP+ FN
    ###
    def forward(self, input, target, smooth=1):
        # TODO check if dim=0 is correct
        input = F.softmax(input, dim=0)

        # input = (input>0.5).float()
        zeros = torch.zeros_like(input)
        ones = torch.ones_like(input)
        torch.where(input > 0.5, ones, zeros)
        input_flat = input.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (input_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + smooth) /
                    (input_flat.sum() + target_flat.sum() + smooth))

        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, input, target):
        CE_loss = self.cross_entropy(input, target)
        dice_loss = self.dice(input, target, self.smooth)
        return (CE_loss + dice_loss) / 2.


def calc_tpr_fpr(input, target, threshold):
    input = (input > threshold).float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    TP = (input_flat * target_flat).sum()
    FP = input_flat.sum() - TP

    input_switched = (input_flat < 0.5).float()
    target_switched = (target_flat < 0.5).float()

    TN = (input_switched * target_switched).sum()
    FN = input_switched.sum() - TN

    # https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
    TPR = TP / (TP + FN + 0.00000000000001)  # Sensitivity, Recall
    FPR = FP / (TN + FP + 0.00000000000001)
    ##TODO Better PRecision/ Recall curve, as data is imbalanced? https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432
    Precision = TP / (TP + FP + 0.00000000000001)

    return TPR, FPR, Precision, threshold


def AUC(input, target):
    input = torch.sigmoid(input.float())

    points = []
    for threshold in np.linspace(0., 1., 20):
        points.append(calc_tpr_fpr(input, target, threshold))

    auc_roc = 0
    auc_pr = 0
    for i, point in enumerate(points):
        if (i < len(points) - 1):
            auc_roc += (points[i][1] - points[i + 1][1]) * (points[i + 1][0] + (points[i][0] - points[i + 1][0]) / 2)
            auc_pr += (point[0] - points[i + 1][0]) * (point[2] + (points[i + 1][2] - point[2]) / 2)

    return auc_roc.numpy(), auc_pr.numpy(), np.array(points)
