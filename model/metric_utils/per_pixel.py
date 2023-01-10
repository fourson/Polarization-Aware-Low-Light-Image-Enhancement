import torch
import torch.jit


@torch.jit.script
def accuracy(pred, target):
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # the accuracy in all pixels
    true = torch.sum(pred == target).double()  # TP + TN
    total = torch.numel(pred)  # TP + TN + FP + FN
    return true / total


@torch.jit.script
def precision(pred, target):
    # precision = TP / (TP + FP)
    # the accuracy in pixels which are predicted as positive
    true_positive = torch.sum(pred * target).double()  # TP
    pred_positive = torch.sum(pred)  # TP + FP
    return true_positive / (pred_positive + 1e-8)


@torch.jit.script
def recall(pred, target):
    # recall = TP / (TP + FN)
    # the accuracy in pixels which are positive
    true_positive = torch.sum(pred * target).double()  # TP
    target_positive = torch.sum(target)  # TP + FN
    return true_positive / (target_positive + 1e-8)


@torch.jit.script
def f1_score(pred, target):
    # f1_score = 2 * precision * recall / (precision + recall)
    p = precision(pred, target)
    r = recall(pred, target)
    if p == 0 and r == 0:
        return 0
    else:
        return 2 * p * r / (p + r)


@torch.jit.script
def MIoU(pred, target):
    # MIoU  = TP / (TP + FN + FP)
    # Mean Intersection over Union of the **positive pixels** (TP + FN) and the **predicted postive pixels** (TP + FP)
    true_positive = torch.sum(pred * target).double()  # TP
    target_positive = torch.sum(target)  # TP + FN
    false_positive = torch.sum(pred * (1 - target))  # FP
    return (true_positive + 1e-8) / (target_positive + false_positive + 1e-8)
