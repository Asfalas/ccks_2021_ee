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
schema = json.load(open('schema.json'))
# print(list(schema.keys()).index('财经/交易-出售/收购'))
event_list = list(schema.keys())
json.dump(event_list, sys.stdout, indent=2, ensure_ascii=False)