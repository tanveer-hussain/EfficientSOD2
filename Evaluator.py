import numpy as np 
from PIL import Image
import cv2
import os
import torch

def eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def f_measure(pred,gt):
    beta2 = 0.3
    with torch.no_grad():
        pred = torch.from_numpy(pred).float().cuda()
        gt = torch.from_numpy(gt).float().cuda()

        prec, recall = eval_pr(pred, gt, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0 # for Nan
    return f_score

def get_metric(sample_batched, result,result_save_path=None,if_recover=True):
    id=sample_batched['meta']['id'][0]
    gt=np.array(Image.open(sample_batched['meta']['gt_path'][0]).convert('L'))/255.0

    if if_recover:
        result=cv2.resize(result, gt.shape[::-1], interpolation=cv2.INTER_LINEAR) 
    else:
        gt=cv2.resize(gt, result.shape[::-1], interpolation=cv2.INTER_NEAREST)

    result=(result*255).astype(np.uint8)

    if result_save_path is not None:
        Image.fromarray(result).save(os.path.join(result_save_path,id+'.png'))

    result=result.astype(np.float64)/255.0

    mae= np.mean(np.abs(result-gt))
    f_score=f_measure(result,gt)
    return mae,f_score

def metric_better_than(metric_a, metric_b):
    if metric_b is None:
        return True
    if isinstance(metric_a,list) or isinstance(metric_a,np.ndarray):
        metric_a,metric_b=metric_a[0],metric_b[0]
    return metric_a < metric_b