import sys
sys.path.append('./')
import json
import logging
import torch
import copy

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce
from module.common.common_dataset import *

class JointDataHandler(CommonDataHandler):
    def __init__(self, path, conf, debug=0):
        self.path = path
        if 'test' in self.path:
            self.data = [line for line in open(path, encoding='utf-8')]
        else:
            self.data = json.load(open(path, encoding='utf-8'))
        logging.info('  debug 模式:' + ("True" if debug else "False"))
        if debug:
            self.data = self.data[:200]
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.role_list = conf.get("role_list", [])
        self.entity_list = conf.get("entity_list", [])
        self.event_list = conf.get("event_list", [])
        self.use_crf = conf.get('use_crf', 0)
        self.conf = conf

    def _load_data(self):
        # for output tensor
        max_len = 0
        tokens_tensor = []
        att_mask_tensor = []
        tri_seq_label_tensor = []
        ent_seq_label_tensor = []

        ent_type_label_tensor = []
        evt_type_label_tensor = []

        tri_seq_mention_mask_tensor = []
        ent_seq_mention_mask_tensor = []
        arg_role_label_tensor = []

        offset_err = 0
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        mask_template = [0.0 for i in range(self.max_seq_len)]

        for info in tqdm(self.data):
            if 'test' in self.path:
                info = json.loads(info)
            tokens = info['text']
            max_len = len(tokens) if len(tokens) > max_len else max_len
            token_ids, att_mask = self.sentence_padding(tokens)

            seq_label_template = [0 for i in range(self.max_seq_len)]
            for idx, ids in enumerate(token_ids):
                if ids == pad_id:
                    seq_label_template[idx] = -100

            tokens_tensor.append(token_ids)
            att_mask_tensor.append(att_mask)

            ent_seq_mention_mask_unit = []
            ent_seq_label = copy.deepcopy(seq_label_template)
            if 'test' not in self.path:
                for ent in info['entity_list']:
                    beg = ent['beg']
                    end = ent['end']
                    if end >= self.max_seq_len:
                        offset_err += 1
                        continue

                    tmp_mask = copy.deepcopy(mask_template)
                    for i in range(beg + 1, end + 1):
                        tmp_mask[i] = 1.0 / (end - beg)
                        ent_seq_label[i] = 2
                    ent_seq_label[beg + 1] -= 1
                    ent_seq_mention_mask_unit.append(tmp_mask)

                ent_seq_mention_mask_unit = ent_seq_mention_mask_unit[:self.max_ent_len]
                for i in range(len(ent_seq_mention_mask_unit), self.max_ent_len):
                    ent_seq_mention_mask_unit.append(copy.deepcopy(mask_template))

                tri_seq_mention_mask_unit = []
                tri_seq_label = copy.deepcopy(seq_label_template)
                arg_role_label_unit = []
                evt_type_label_unit = []

                for evt in info.get('event_list', []):
                    evt_type = evt['event_type']
                    beg = evt['beg']
                    end = evt['end']
                    if end >= self.max_seq_len:
                        offset_err += 1
                        continue
                    tmp_mask = copy.deepcopy(mask_template)
                    for i in range(beg+1, end+1):
                        tmp_mask[i] = 1.0 / (end - beg)
                        tri_seq_label[i] = self.event_list.index(evt_type) * 2 + 2
                    tri_seq_label[beg+1] -= 1
                    tri_seq_mention_mask_unit.append(tmp_mask)
                    evt_type_label_unit.append(self.event_list.index(evt_type))

                    arg_role_map = {}
                    for a in evt['argument_list']:
                        beg = a['beg']
                        end = a['end']
                        mention = info['text'][beg: end]
                        role = a['argument_role']
                        arg_role_map[str(beg) + '@#@' + mention] = role

                    tmp_role_label = []
                    for ent in info['entity_list']:
                        ent_str = str(ent['beg']) + '@#@' + info['text'][ent['beg']: ent['end']]
                        if ent_str in arg_role_map:
                            tmp_role_label.append(self.role_list.index(arg_role_map[ent_str]))
                        else:
                            tmp_role_label.append(0)
                    tmp_role_label = tmp_role_label[:self.max_ent_len]
                    for i in range(len(tmp_role_label), self.max_ent_len):
                        tmp_role_label.append(-100)
                    arg_role_label_unit.append(tmp_role_label)

                tri_seq_mention_mask_unit = tri_seq_mention_mask_unit[:self.max_evt_len]
                arg_role_label_unit = arg_role_label_unit[:self.max_evt_len]
                evt_type_label_unit = evt_type_label_unit[:self.max_evt_len]
                for i in range(len(tri_seq_mention_mask_unit), self.max_evt_len):
                    tri_seq_mention_mask_unit.append(copy.deepcopy(mask_template))
                    arg_role_label_unit.append([-100] * self.max_ent_len)
                    evt_type_label_unit.append(0)
            if 'test' not in self.path:
                tri_seq_label_tensor.append(tri_seq_label)
                ent_seq_label_tensor.append(ent_seq_label)
                arg_role_label_tensor.append(arg_role_label_unit)
                tri_seq_mention_mask_tensor.append(tri_seq_mention_mask_unit)
                ent_seq_mention_mask_tensor.append(ent_seq_mention_mask_unit)
                evt_type_label_tensor.append(evt_type_label_unit)
                # print(tokens)
                # print(tri_seq_label)
                # print(ent_seq_label)
                # print(arg_role_label_tensor)
                # print(tri_seq_mention_mask_tensor)
                # print(ent_seq_mention_mask_tensor)
                # input()

        # transform to tensor
        tokens_tensor = torch.LongTensor(tokens_tensor)
        att_mask_tensor = torch.ByteTensor(att_mask_tensor)
        if 'test' not in self.path:
            tri_seq_label_tensor = torch.LongTensor(tri_seq_label_tensor)
            ent_seq_label_tensor = torch.LongTensor(ent_seq_label_tensor)
            tri_seq_mention_mask_tensor = torch.FloatTensor(tri_seq_mention_mask_tensor)
            ent_seq_mention_mask_tensor = torch.FloatTensor(ent_seq_mention_mask_tensor)
            arg_role_label_tensor = torch.LongTensor(arg_role_label_tensor)
            evt_type_label_tensor = torch.LongTensor(evt_type_label_tensor)

        if offset_err:
            logging.warn("  越界: " + str(offset_err))
        if 'test' not in self.path:
            return tokens_tensor, att_mask_tensor, tri_seq_mention_mask_tensor, ent_seq_mention_mask_tensor, evt_type_label_tensor, tri_seq_label_tensor, ent_seq_label_tensor, arg_role_label_tensor
        return tokens_tensor, att_mask_tensor

