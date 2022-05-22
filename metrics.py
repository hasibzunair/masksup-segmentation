import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, jaccard_score

def iou_score(output, target):
    # Taken from https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/metrics.py
    smooth = 1e-30

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    # https://stackoverflow.com/questions/61488732/how-calculate-the-dice-coefficient-for-multi-class-segmentation-task-using-python
    
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    y_true = output_
    y_pred = target_ 
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1e-30
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


# def dice_coef(y_true, y_pred):
#     # https://stackoverflow.com/questions/61488732/how-calculate-the-dice-coefficient-for-multi-class-segmentation-task-using-pytho
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     smooth = 0.0001
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


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
        return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

    dice = dice_coef(gt, pred)
    
    #import ipdb; ipdb.set_trace()
    AUC = roc_auc_score(gt.flatten(), pred.flatten())
    jc = jaccard_score(gt.flatten(), pred.flatten())
    return dice, jc, AUC