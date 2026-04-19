import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def KLConsistencyLoss(output, pred_label, args, temperature=2):
    """
    Cross-source KL-divergence regularization loss (Eq.29 and Eq.31 in paper).

    For each pair of source models (m, n), enforces consistency between their
    predictive distributions over target samples by minimizing the symmetric KL divergence:

        L_KL = (1/|m,n|) * sum_{m,n} [ D_KL(p_m || p_n) + D_KL(p_n || p_m) ] / 2

    This is computed class-wise (averaged over pseudo-label classes) to account for
    the per-class distribution alignment described in Section 3.5.

    Args:
        output: [N_src, B, K] weighted source logit outputs (source x batch x class)
        pred_label: [B] pseudo-labels for the current batch
        args: argument namespace with class_num and src
        temperature: softmax temperature (unused here; kept for API compatibility)
    """
    eps = 1e-16
    KL_loss = 0
    num_pairs = 0

    label_id = pred_label.cpu().numpy()
    label_id = np.unique(label_id)

    for cls in range(args.class_num):
        if cls in label_id:
            prob_cls_all = torch.ones(len(args.src), args.class_num)

            for i in range(len(args.src)):
                mask_cls = pred_label.cpu() == cls
                mask_cls_ex = torch.repeat_interleave(mask_cls.unsqueeze(1), args.class_num, dim=1)
                logits_cls = torch.sum(output[i] * mask_cls_ex.float(), dim=0)
                cls_num = torch.sum(mask_cls)
                logits_cls_acti = logits_cls * 1.0 / (cls_num + eps)
                prob_cls = torch.softmax(logits_cls_acti, dim=0)
                prob_cls = torch.clamp(prob_cls, 1e-8, 1.0)
                prob_cls_all[i] = prob_cls

            # Symmetric KL divergence for each ordered pair (m, n) -- Eq.29
            for m in range(len(args.src)):
                for n in range(len(args.src)):
                    if m != n:
                        KL_div = (
                            torch.sum(prob_cls_all[m] * torch.log(prob_cls_all[m] / prob_cls_all[n])) +
                            torch.sum(prob_cls_all[n] * torch.log(prob_cls_all[n] / prob_cls_all[m]))
                        ) / 2.0
                        KL_loss += KL_div
                        num_pairs += 1

    if num_pairs > 0:
        KL_loss = KL_loss / num_pairs
    return KL_loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes #标签平滑
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss



def MSEConsistencyLoss(output, pred_label, args):
    """
    Cross-source L2-style matching loss (Eq.30 and Eq.31 in paper).

    For each pair of source models (m, n), reduces the discrepancy between their
    output distributions in value space:

        L_MSE = (1/K) * sum_k |p_m(y_k|x_i) - p_n(y_k|x_i)|^2

    Averaged over all unique source pairs and pseudo-label classes (Section 3.5).

    Args:
        output: [N_src, B, K] weighted source logit outputs
        pred_label: [B] pseudo-labels for the current batch
        args: argument namespace with class_num and src
    """
    eps = 1e-16
    MSE_loss = 0
    num_pairs = 0

    label_id = pred_label.cpu().numpy()
    label_id = np.unique(label_id)

    for cls in range(args.class_num):
        if cls in label_id:
            prob_cls_all = torch.ones(len(args.src), args.class_num)

            for i in range(len(args.src)):
                mask_cls = pred_label.cpu() == cls
                mask_cls_ex = torch.repeat_interleave(mask_cls.unsqueeze(1), args.class_num, dim=1)
                logits_cls = torch.sum(output[i] * mask_cls_ex.float(), dim=0)
                cls_num = torch.sum(mask_cls)
                logits_cls_acti = logits_cls * 1.0 / (cls_num + eps)
                prob_cls = torch.softmax(logits_cls_acti, dim=0)
                prob_cls = torch.clamp(prob_cls, 1e-8, 1.0)
                prob_cls_all[i] = prob_cls

            # L2 matching for each unique pair (m, n) with m < n -- Eq.30
            for m in range(len(args.src)):
                for n in range(m + 1, len(args.src)):
                    # (1/K) * sum_k |p_m(y_k|x_i) - p_n(y_k|x_i)|^2
                    MSE_div = F.mse_loss(prob_cls_all[m], prob_cls_all[n])
                    MSE_loss += MSE_div
                    num_pairs += 1

    if num_pairs > 0:
        MSE_loss = MSE_loss / num_pairs
    return MSE_loss
