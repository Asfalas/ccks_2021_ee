import json
import sys

from tqdm import tqdm

# 统计
data = [line for line in open('./duee_train.json')] + [line for line in open('./duee_dev.json')]

max_seq_len = 0
max_ent_len = 0
max_role_len = max([len(v) for v in json.load(open('./schema.json')).values()])
evt_num = len(json.load(open('./schema.json')).keys())

for line in tqdm(data):
    d = json.loads(line)
    ent_set = set()
    max_seq_len = len(d['text']) if len(d['text']) > max_seq_len else max_seq_len
    for e in d['event_list']:
        for a in e['arguments']:
            ent_set.add(str(a['argument_start_index']) + '-' + a['argument'])
    max_ent_len = max_ent_len if max_ent_len > len(ent_set) else len(ent_set)

print(max_seq_len)
print(max_ent_len)
print(max_role_len)
print(evt_num)

# 转化
convert_map = {
    './duee_train.json': "./duee_cls_train.json",
    './duee_dev.json': "./duee_cls_dev.json"
}

for in_file, out_file in convert_map.items():
    out = []
    data = [line for line in open(in_file)]
    for line in tqdm(data):
        d = json.loads(line)
        ent_set = set()
        for e in d['event_list']:
            for a in e['arguments']:
                ent_set.add('@#@'.join([str(a['argument_start_index']), a['argument']]))
        
        for e in d['event_list']:
            tmp = {
                'text': d['text'],
                'id': d['id'],
                'event': {
                    "event_type": e['event_type'],
                    "trigger": e['trigger'],
                    "trigger_start_index": e['trigger_start_index'],
                } 
            }
            tmp_args = []
            arg_map = {}
            for a in e['arguments']:
                arg_map['@#@'.join([str(a['argument_start_index']), a['argument']])] = a['role']
            for ent in ent_set:
                argument_start_index, argument = ent.split('@#@')
                role = 'None'
                if ent in arg_map:
                    role = arg_map[ent]
                tmp_args.append({
                    'argument_start_index': int(argument_start_index),
                    'argument': argument,
                    'role': role
                })
            tmp['event']['argument'] = tmp_args
            out.append(tmp)
    json.dump(out, open(out_file, 'w'), indent=2, ensure_ascii=False)
                


