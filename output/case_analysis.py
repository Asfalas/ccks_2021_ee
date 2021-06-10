import json

def generate_cases(pred_path, ground_truth_path):
    pred_items = [json.loads(line) for line in open(pred_path)]
    ground_truth_items = json.load(open(ground_truth_path))
    
    output = open("output/case.json", 'w')
    for p, gt in zip(pred_items, ground_truth_items):
        assert p['text'] == gt['text']
        pe = set()
        pa = set()
        for e in p.get("event_list", []):
            e_key = tuple(e['trigger'])
            pe.add(e_key)
            for a in e['argument']:
                pa.add(tuple(a) + tuple(e_key))
        ge = set()
        ga = set()
        tmp_event_list = []
        for e in gt.get("event_list", []):
            e_key = (e['event_type'], int(e['beg']), e['event_content'])
            ge.add(e_key)
            tmp_argument_list = []
            for a in e['argument_list']:
                ga.add((a['argument_role'], int(a['beg']), a['argument_content']) + e_key)
                tmp_argument_list.append([a['argument_role'], int(a['beg']), a['argument_content']])
            tmp_event_list.append({
                'trigger': list(e_key),
                'argument': tmp_argument_list
            })
        gt['event_list'] = tmp_event_list
        gt.pop('entity_list')
        pred_num = len(pe) + len(pa)
        true_num = len(pe & ge) + len(pa & ga)
        gt_num = len(ge) + len(ga)
        
        pre = float(true_num / pred_num) if pred_num != 0 else 0
        r = float(true_num / gt_num) if gt_num != 0 else 0
        f1 = 2 * pre * r / (pre + r) if pre + r != 0 else 0
        
        if f1 < 0.9:
            output.write(str(f'{str(round(f1, 3))}, true_num: {str(true_num)}, pred_num: {str(pred_num)}, gt_num: {str(gt_num)}') + '\n')
            output.write("预测：" + json.dumps(p, ensure_ascii=False) + '\n')
            output.write("答案：" + json.dumps(gt, ensure_ascii=False) + '\n')
            output.write('\n')
    output.close()
            

if __name__ == "__main__":
    generate_cases('output/joint_eval.json', 'data/tmp_dev.json')