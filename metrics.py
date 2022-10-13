import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, jaccard_score


def calculate_metric_percase(pred, gt):

    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy() > 0.5
    if torch.is_tensor(gt):
        gt = gt.data.cpu().numpy()

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = np.sum(y_true * y_pred)
        return (2.0 * intersection + smooth) / (
            np.sum(y_true) + np.sum(y_pred) + smooth
        )

    dice = dice_coef(gt, pred)

    # import ipdb; ipdb.set_trace()
    AUC = roc_auc_score(gt.flatten(), pred.flatten())
    jc = jaccard_score(gt.flatten(), pred.flatten())
    return dice, jc, AUC


# Taken from https://github.com/McGregorWwww/UCTransNet/blob/e82df91015700e80fe67cea3246f2439be36166f/utils.py
def iou_on_batch(masks, pred):
    """Computes the mean Area Under ROC Curve over a batch during training"""
    ious = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        np.sum(y_true_f) + np.sum(y_pred_f) + smooth
    )


def dice_on_batch(masks, pred):
    """Computes the mean Area Under ROC Curve over a batch during training"""
    dices = []

    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        # print("www",np.max(prediction), np.min(prediction))
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        # print("2",np.sum(tmp))
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        # print("rrr",np.max(mask), np.min(mask))
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)
