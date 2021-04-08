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
max_len = 0
for line in data:
    d = json.loads(line)
    max_len = max_len if max_len > len(d['event_list']) else len(d['event_list'])
print(max_len)