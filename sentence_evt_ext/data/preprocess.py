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
#
# schema = json.load(open("schema.json"))
# role_list = ['None']
# for k, v in schema.items():
#     if k == 'None':
#         continue
#     for a in v:
#         if a == 'None':
#             continue
#         role_list.append(k + "@#@" + a)
# json.dump(role_list, sys.stdout, indent=2, ensure_ascii=False)
#


data = [line for line in open("duee_train.json", encoding='utf-8')]
new_data = []
for line in data:
    d = json.loads(line)
    arg_set = set()
    for e in d['event_list']:
        for a in e['arguments']:
            arg_set.add(str(a['argument_start_index']) + "@#@" + str(a['argument']))
    d['ent_list'] = list(arg_set)
    new_data.append(d)
json.dump(new_data, open("./duee_joint_train.json", 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

data = [line for line in open("duee_dev.json", encoding='utf-8')]
new_data = []
for line in data:
    d = json.loads(line)
    arg_set = set()
    for e in d['event_list']:
        for a in e['arguments']:
            arg_set.add(str(a['argument_start_index']) + "@#@" + str(a['argument']))
    d['ent_list'] = list(arg_set)
    new_data.append(d)
json.dump(new_data, open("./duee_joint_dev.json", 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


# data = [line for line in open("duee_train.json")] + [line for line in open("duee_dev.json")]
# evt_len = []
# for line in data:
#     d = json.loads(line)
#     evt_len.append(len(d['event_list']))
# evt_len = sorted(evt_len)
# print("avg: " + str(sum(evt_len)/len(evt_len)))
# print("max: " + str(evt_len[-1]))
# print("pct90: " + str(evt_len[int(0.9 * len(evt_len))]))
# print("pct99: " + str(evt_len[int(0.99 * len(evt_len))]))


# total = 0
# over = 0
# total_map = {}
# end_map = {}
# for line in data:
#     overlap_map = {}
#     d = json.loads(line)
#     for e in d['event_list']:
#         et = e['event_type']
#         for a in e['arguments']:
#             beg = str(a['argument_start_index'])
#             argument = a['argument']
# #             role = et + '@#@' + a['role']
#             if a['role'] == '时间':
#                 continue
#             key = beg + '@#@' + argument
#             if key not in overlap_map:
#                 overlap_map[key] = []
#             overlap_map[key].append(et)
#             total += 1

#     for k, v in overlap_map.items():
#         if len(v) > 1:
#             over += len(v)
#             sorted(v)
#             for i in range(len(v)-1):
#                 for j in range(i, len(v)):
#                     if v[i] not in total_map:
#                         total_map[v[i]] = set()
#                     total_map[v[i]].add(v[j])

#                     if v[j] not in end_map:
#                         end_map[v[j]] = set()
#                     end_map[v[j]].add(v[i])
# total_map.update(end_map)
# set_list = [set()]

# for i, j in total_map.items():
#     is_fit = False
#     for s in set_list:
#         conflict = False
#         for k in j:
#             if k in s:
#                 conflict = True
#             break
#         if not conflict:
#             s.add(i)
#             is_fit = True
#             break
#         else:
#             continue
#     if not is_fit:
#         set_list.append(set())
#         set_list[-1].add(i)

# for s in set_list:
#     print(s)

# print(over)
# print(total)


# schema = json.load(open("schema.json"))
# for k, v in schema.items():
#     if k=='None':
#         continue
#     if k in set_list[0] and k in set_list[1]:
#         raise Exception('error')
#     if k in set_list[0] or k in set_list[1]:
#         continue
#     if len(set_list[0]) < len(set_list[1]):
#         set_list[0].add(k)
#     else:
#         set_list[1].add(k)
#     for a in v:
#         if a == 'None':
#             continue
#         t = k+'@#@'+a
#         if '时间' in t:
#             if '时间' not in label_list1:
#                 label_list1.append('时间')
#             if '时间' not in label_list2:
#                 label_list2.append('时间')
#         else:
#             if t not in set_list[1] and t not in set_list[0]:
#                 label_list1.append(t)
#                 label_list2.append(t)
#             elif t in set_list[0]:
#                 label_list1.append(t)
#             else:
#                 label_list2.append(t)
# res = {
#     'label1': list(set_list[0]),
#     'label2': list(set_list[1])
# }

# json.dump(res, open("./joint_evt_schema.json", 'w'), indent=2, ensure_ascii=False)

            
