import numpy as np

def pixel_accuracy(pred, gt):
    """
    Calculates the pixel-wise accuracy between two masks.
    """
    return (pred == gt).sum() / np.prod(pred.shape)

def dice_coefficient(pred, gt, smooth=1e-6):
    """
    Calculates the Dice coefficient (F1-Score) between two binary masks.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def iou(pred, gt, smooth=1e-6):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return (intersection + smooth) / (union + smooth)


