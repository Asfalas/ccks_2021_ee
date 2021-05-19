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

def evaluate_per_classes(ground_truth, pred, labels):
    max_len = len(ground_truth)
    pred_map = {}
    def get_res_map(l):
        index = 0
        m = {}
        while index < max_len:
            if labels[l[index]][0] == 'B':
                event_type = labels[l[index]][2:]
                beg = index
                index += 1
                while(index < max_len and labels[l[index]] == 'I-' + event_type):
                    index += 1
                end = index
                if event_type not in m:
                    m[event_type] = set()
                m[event_type].add(str(beg)+'-'+str(end))
            else:
                index += 1
        return m
    ground_truth_map = get_res_map(ground_truth)
    pred_map = get_res_map(pred)
    pres = []
    recalls = []

    iters = []
    events = [line.strip() for line in open('data/output_event_order_list.txt').readlines()]
    if len(set(events) & set(ground_truth_map.keys())) > 0:
        iters = events
    else:
        iters = ground_truth_map.keys()
    for k in iters:
        if k not in ground_truth_map and k not in pred_map:
            print(f"类型: {k}, 精确率: {'-'} ({'-'}/{'-'}), 召回率: {'-'} ({'-'}/{'-'})")
            continue
        precision = 0.0
        pred_count = 0
        right_count = 0
        total_count = 0
        if k not in pred_map:
            precision = 0.0
            pred_count = 0
            right_count = 0
            total_count = len(ground_truth_map[k])
            recall = 0
        elif k not in ground_truth_map:
            precision = 0.0
            pred_count = len(pred_map[k])
            right_count = 0
            total_count = 0
            recall = 0
        else:
            pred_count = len(pred_map[k])
            total_count = len(ground_truth_map[k])
            right_count = len(ground_truth_map[k] & pred_map[k])
            precision = round(float(right_count) / pred_count, 4)
            recall = round(float(right_count) / total_count, 4)
        if pred_count != 0:
            pres.append(precision)
        if total_count != 0:
            recalls.append(recall)
        print(f"类型: {k}, 精确率: {precision} ({right_count}/{pred_count}), 召回率: {recall} ({right_count}/{total_count})")
    print("平均精确率:" + str(np.mean(np.array(pres))))
    print("平均召回率:" + str(np.mean(np.array(recalls))))
