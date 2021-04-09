import json
import sys
# def schema_process():
#     schema = open("duee_event_schema.json")
#     evt_schema = {}
#     for line in schema:
#         d = json.loads(line)
#         evt_schema[d['event_type']] = ['None']
#         for role in d['role_list']:
#             evt_schema[d['event_type']].append(role['role'])

#     json.dump(evt_schema, open('schema.json', 'w'), indent=2, ensure_ascii=False)
# schema = json.load(open('schema.json'))
# # print(list(schema.keys()).index('财经/交易-出售/收购'))
# event_list = list(schema.keys())
# json.dump(event_list, sys.stdout, indent=2, ensure_ascii=False)

data = [line for line in open("duee_train.json")] + [line for line in open("duee_dev.json")]
# max_len = 0
# for line in data:
#     d = json.loads(line)
#     max_len = max_len if max_len > len(d['event_list']) else len(d['event_list'])
# print(max_len)

total = 0
over = 0
total_map = {}
end_map = {}
for line in data:
    overlap_map = {}
    d = json.loads(line)
    for e in d['event_list']:
        et = e['event_type']
        for a in e['arguments']:
            beg = str(a['argument_start_index'])
            argument = a['argument']
            role = et + '@#@' + a['role']
            key = beg + '@#@' + argument
            if key not in overlap_map:
                overlap_map[key] = []
            overlap_map[key].append(role)
            total += 1

    for k, v in overlap_map.items():
        if len(v) > 1:
            over += len(v)
            sorted(v)
            for i in range(len(v)-1):
                for j in range(i, len(v)):
                    if v[j] == v[i] or '时间' in v[i] or '时间' in v[j]:
                        continue
                    if v[i] not in total_map:
                        total_map[v[i]] = set()
                    total_map[v[i]].add(v[j])

                    if v[j] not in end_map:
                        end_map[v[j]] = set()
                    end_map[v[j]].add(v[i])
total_map.update(end_map)
set_list = [set()]

for i, j in total_map.items():
    is_fit = False
    for s in set_list:
        conflict = False
        for k in j:
            if k in s:
                conflict = True
            break
        if not conflict:
            s.add(i)
            is_fit = True
            break
        else:
            continue
    if not is_fit:
        set_list.append(set())
        set_list[-1].add(i)

for s in set_list:
    print(s)

print(over)
print(total)


schema = json.load(open("schema.json"))
label_list1 = []
label_list2 = []
for k, v in schema.items():
    if k=='None':
        continue
    for a in v:
        if a == 'None':
            continue
        t = k+'@#@'+a
        if '时间' in t:
            if '时间' not in label_list1:
                label_list1.append('时间')
            if '时间' not in label_list2:
                label_list2.append('时间')
        else:
            if t not in set_list[1] and t not in set_list[0]:
                label_list1.append(t)
                label_list2.append(t)
            elif t in set_list[0]:
                label_list1.append(t)
            else:
                label_list2.append(t)
res = {
    'label1': label_list1,
    'label2': label_list2
}

json.dump(res, open("./joint_schema.json", 'w'), indent=2, ensure_ascii=False)

            
