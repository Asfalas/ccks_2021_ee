import json
import sys

def generate_possible_role_list():
    data = json.load(open('./data/event_schema.json'))
    role_list = ['None']
    for k, v in data.items():
        for role in v:
            role_list.append(k+"@#@"+role)
    json.dump(role_list, open('data/possible_role_list.json', 'w'), ensure_ascii=False, indent=2)

def generate_role_list():
    data = json.load(open('./data/event_schema.json'))
    role_list = set()
    for k, v in data.items():
        for role in v:
            role_list.add(role)
    json.dump(['None'] + list(role_list), open('data/role_list.json', 'w'), ensure_ascii=False, indent=2)

def generate_event_schema():
    data = json.load(open('./data/event_schema.json'))
    event_schema = {'None': 'None'}
    for k, v in data.items():
        v  = ['None'] + v
        event_schema[k] = v
    json.dump(event_schema, open('data/schema.json', 'w'), ensure_ascii=False, indent=2)
    json.dump(list(event_schema.keys()), open('data/event_list.json', 'w'), ensure_ascii=False, indent=2)

def generate_source_event_schema():
    data = open('data/train.json')
    event_schema = {}
    for info in data:
        info = json.loads(info)
        event_list = info.get('event_list', [])
        for e_info in event_list:
            et = e_info['trigger'][0]
            if et not in event_schema:
                event_schema[et] = set()
            for a in e_info['argument']:
                event_schema[et].add(a[0])
    for e, a in event_schema.items():
        event_schema[e] = list(a)
    json.dump(event_schema, open('data/event_schema.json', 'w'), indent=2, ensure_ascii=False)

def transform_format():
    data = open('data/train.json')
    res = []
    for info in data:
        info = json.loads(info)
        text = info['text']
        event_list = info.get('event_list', [])
        
        new_entity_list = []
        
        new_event_list = []
        for e in event_list:
            t = e['trigger']
            et, beg, content = t[0], t[1], t[2]
            end = beg + len(content)
            arguments = e['argument']

            new_argument_list = []
            for a in arguments:
                role, a_beg, a_content = a[0], a[1], a[2]
                if role[0] == 'o':
                    role = 'O' + role[1:]
                new_argument_list.append({
                    "argument_role": role,
                    "argument_content": a_content,
                    "beg": a_beg,
                    "end": a_beg + len(a_content)
                })
                new_entity_list.append({
                    "entity_type": "None",
                    "entity_content": a_content,
                    "beg": a_beg,
                    "end": a_beg + len(a_content)
                })

            new_event_list.append({
                'event_type': et,
                "event_content": content,
                "beg": beg,
                "end": end,
                "argument_list": new_argument_list
            })
        res.append({
            "text": text,
            "entity_list": new_entity_list,
            "event_list": new_event_list
        })
    json.dump(res, open('data/format_train.json', 'w'), indent=2, ensure_ascii=False)

def eda():
    data = json.load(open('data/format_train.json'))
    text_lens = []
    event_lens = []
    ent_lens = []
    for info in data:
        text = info['text']
        text_lens.append(len(text))
        event_lens.append(len(info.get("event_list", [])))
        ent_lens.append(len(info.get("entity_list", [])))
        
    for lens in [text_lens, event_lens, ent_lens]:
        lens = sorted(lens)
        print("total: ", len(lens))
        print("max: ", max(lens))
        print("min: ", min(lens))
        print('avg: ', sum(lens)/len(lens))
        print('pct80: ', lens[int(len(lens) * 0.8)])
        print('pct90: ', lens[int(len(lens) * 0.9)])
        print('pct95: ', lens[int(len(lens) * 0.95)])
        print('pct99: ', lens[int(len(lens) * 0.99)])
        print()


if __name__ == "__main__":
    # generate_source_event_schema()
    generate_event_schema()
    generate_role_list()
    generate_possible_role_list()
    eda()

    # transform_format()