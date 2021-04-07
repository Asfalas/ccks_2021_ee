import json
import jsonlines

def generate_test_data():
    evt_info = json.load(open('evt_men_test.json'))
    arg_info = json.load(open('arg_men_test.json'))

    res = []
    max_arg_len = 0
    for evt, arg in zip(evt_info, arg_info):
        if not evt['mention']:
            continue
        for e in evt['mention']:
            tmp = {
                "text": evt['text'],
                "id": evt['id'],
                "event": {}
            }
            trigger_start_index, trigger = e.split('@#@')[0], e.split('@#@')[1]
            tmp['event']['event_type'] = 'None'
            tmp['event']['trigger_start_index'] = int(trigger_start_index)
            tmp['event']['trigger'] = trigger
            tmp['event']['argument'] = []
            max_arg_len = max_arg_len if max_arg_len > len(arg['mention']) else len(arg['mention'])
            for a in arg['mention']:
                argument_start_index, argument = a.split('@#@')[0], a.split('@#@')[1]
                tmp['event']['argument'].append({
                    "argument": argument,
                    "argument_start_index": int(argument_start_index),
                    "role": 'None'
                })
            res.append(tmp)
    json.dump(res, open('../data/duee_cls_test.json', 'w'), indent=2, ensure_ascii=False)
    print(max_arg_len)

def generate_submit_data():
    m = {}
    data = json.load(open('men_cls_test.json'))
    for i in data:
        id = i['id']
        if not i['event']['argument']:
            continue
            
        if id not in m:
            m[id] = []
        m[id].append({
            "event_type": i['event']['event_type'],
            "arguments":[
                {"role": j['role'], "argument": j["argument"]} for j in i['event']['argument']
            ]
        })
    res = []
    for id, e in m.items():
        res.append({
            "id": id,
            "event_list": e
        })
#     json.dump(res, open("./duee.json", 'w'), indent=2, ensure_ascii=False)
    with jsonlines.open("./duee.json", mode='w') as writer:
        for i in res:
            writer.write(i)

    
if __name__ == "__main__":
    generate_submit_data()
        