import json

pred_list = json.load(open("output/joint_test.json"))

gt_list = json.load(open("data/anno_sentence_for_test_total.json"))[:len(pred_list)]


def print_res(m):
    precisions, recalls, f1s = [], [], []
    for e, v in m.items():
        pred_num, true_num, total_num = v["pred_num"], v["true_num"], v["total_num"]
        precision = round(float(true_num/pred_num) if pred_num != 0 else 0, 4)
        recall = round(float(true_num/total_num) if total_num != 0 else 0, 4)
        f1 = round(2 * precision * recall / (precision + recall) if precision + recall != 0 else 0, 4)
        if total_num == 0:
            print(f"{e}:    精确率: - (-/-)   召回率: - (-/-) ")
            continue            
        print(f"{e}:    精确率: {str(precision)} ({true_num}/{pred_num})   召回率: {str(recall)} ({true_num}/{total_num})")
        if e != "micro":
            if precision != 0:
                precisions.append(precision)
            if recall != 0:
                recalls.append(recall)
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    print(f"macro:    精确率: {str(p)}    召回率: {str(r)}    F1: {str(f1)}")

def get_ent_or_evt_metrics(type='ent'):
    entity_list = json.load(open("data/entity_list.json"))
    if type == 'evt':
        entity_list = [l.strip() for l in open("data/output_event_order_list.txt")]

    metrics_map = {}
    for e in entity_list:
        metrics_map[e] = {"pred_num": 0, "true_num": 0, "total_num": 0}

    pred_num, true_num, total_num = 0, 0, 0
    for p, gt in zip(pred_list, gt_list):
        assert p['text'] == gt['text']
        pred_set = set()
        gt_set = set()
        iter_list = 'entity_list' if type == 'ent' else 'event_list'
        for ent in p.get(iter_list, []):
            if type == 'ent':
                pred_set.add('@#@'.join([ent['entity_content'], str(ent['beg']), ent['entity_type']]))
            elif type == 'evt':
                pred_set.add('@#@'.join([ent['event_content'], str(ent['beg']), ent['event_type']]))
        for ent in gt.get(iter_list, []):
            if type == 'ent':
                gt_set.add('@#@'.join([ent['entity_content'], str(ent['beg']), ent['entity_type']]))
            elif type == 'evt':
                gt_set.add('@#@'.join([ent['event_content'], str(ent['beg']), ent['event_type']]))
        true_set = pred_set & gt_set
        pred_num += len(pred_set)
        true_num += len(true_set)
        total_num += len(gt_set)

        for e in entity_list:
            metrics_map[e]['pred_num'] += sum([1 if et.split('@#@')[-1] == e else 0 for et in pred_set])
            metrics_map[e]['true_num'] += sum([1 if et.split('@#@')[-1] == e else 0 for et in true_set])
            metrics_map[e]['total_num'] += sum([1 if et.split('@#@')[-1] == e else 0 for et in gt_set])
        metrics_map["micro"] = {"pred_num": pred_num, "true_num": true_num, "total_num": total_num}
    print_res(metrics_map)
    print("")
    return metrics_map


if __name__ == "__main__":
    m = get_ent_or_evt_metrics('ent')
    m = get_ent_or_evt_metrics('evt')