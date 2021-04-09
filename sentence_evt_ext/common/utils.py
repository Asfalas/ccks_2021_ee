import sys
sys.path.append('./')

import torch
import numpy as np
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_metrics(ground_truth, pred_label, labels, average='micro'):
    precision = precision_score(ground_truth, pred_label, labels=labels, average=average, zero_division=0)
    recall = recall_score(ground_truth, pred_label, labels=labels, average=average, zero_division=0)
    # f1 = f1_score(ground_truth, pred_label, average=average, zero_division=0)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1