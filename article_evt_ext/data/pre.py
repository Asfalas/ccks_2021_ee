import json

def statistics():
    data = [line for line in open("article_evt_ext/data/duee_fin_train.json")] + [line for line in open("article_evt_ext/data/duee_fin_dev.json")]
    total = 0
    c = 0
    text_lens = []
    for line in data:
        d = json.loads(line)
        text = d.get("text", '')
        # event_list = d.get("event_list", [])
        # text_lens.append(len(event_list))
        # text = d.get("text", '')[:768]
        c = 0
        arg_set = set()
        for e in d.get('event_list', []):
            for a in e.get("arguments", []):
                arg_set.add(a['argument'])
        text_lens.append(len(arg_set))


        # text_lens.append(len(d.get('event_list', [])))
    text_lens = sorted(text_lens)
    print("avg: " + str(sum(text_lens) / len(text_lens)))
    print("max: " + str(text_lens[-1]))
    print("min: " + str(text_lens[0]))
    print("pct95: " + str(text_lens[int(0.95 * len(text_lens))]))
    print("pct90: " + str(text_lens[int(0.9 * len(text_lens))]))
    print("pct99: " + str(text_lens[int(0.99 * len(text_lens))]))


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        assert s[index: index + len(sub)] == sub
        index_list.append(index)
        index = s.find(sub,index+1)
    
    if len(index_list) > 0:
        return index_list
    else:
        return []


def generate_training_data(input_file="article_evt_ext/data/duee_fin_train.json", output_file="article_evt_ext/data/duee_fin_joint_train.json"):
    data = [line for line in open(input_file)]
    new_data = []
    for line in data:
        d = json.loads(line)
        text = d.get("text", '')
        event_list = d.get("event_list", [])
        new_event_list = []
        ent_set = set()
        enum = 'None'
        for e in event_list:
            trigger = e["trigger"]
            indexs = find_all(trigger, text)
            if not indexs:
                continue
            new_e = {
                "event_type": e['event_type'],
                "trigger": e["trigger"],
                "indexs": indexs,
                "arguments": []
            }
            new_arguments = []
            for a in e.get("arguments"):
                argument = a['argument']
                ent_set.add(argument)
                if a['role'] == "环节":
                    enum = argument
                    continue
                indexs = find_all(argument, text)
                if not indexs:
                    continue
                new_a = {
                    "argument": argument,
                    "role": a['role'],
                    "indexs": indexs
                }
                new_arguments.append(new_a)
            new_e['arguments'] = new_arguments
            new_event_list.append(new_e)
        new_ent_list = []
        for e in ent_set:
            indexs = find_all(e, text)
            if not indexs:
                continue
            new_ent_list.append({
                "ent": e,
                "indexs": indexs
            })
        new_d = {
            "text": text,
            "id": d['id'],
            "event_list": new_event_list,
            "ent_list": new_ent_list,
            "enum": enum
        }
        new_data.append(new_d)
    json.dump(new_data, open(output_file, 'w'), indent=2, ensure_ascii=False)
            

def generate_role_list():
    data = [line for line in open("article_evt_ext/data/duee_fin_event_schema.json")]
    role_list = ['None']
    for line in data:
        d = json.loads(line)
        et = d['event_type']
        for r in d['role_list']:
            role_list.append(et+"@#@"+r['role'])
    json.dump(role_list, open("article_evt_ext/data/duee_fin_role_list.json", 'w'), indent=2, ensure_ascii=False)

def generate_multi_tagger_label_list():
    data = [line for line in open("article_evt_ext/data/duee_fin_event_schema.json")]
    res_map = {}
    for line in data:
        d = json.loads(line)
        et = d['event_type']
        res_map[et] = []
        for r in d['role_list']:
            if r['role'] == "环节":
                continue
            res_map[et].append(r['role'])
    json.dump(res_map, open("article_evt_ext/data/duee_fin_label_map.json", 'w'), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # statistics()
#     generate_role_list()
    # generate_training_data()
    # generate_training_data("article_evt_ext/data/duee_fin_dev.json", "article_evt_ext/data/duee_fin_joint_dev.json")
    generate_multi_tagger_label_list()

