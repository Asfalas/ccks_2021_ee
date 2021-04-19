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
from sub_module.common.common_seq_tag_dataset import *

class JointDataHandler(CommonSeqTagDataHandler):
    def __init__(self, path, conf, debug=0):
        self.path = path
        if 'test' in path:
            self.data = [line for line in open(path, encoding='utf-8').readlines()]
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
        self.use_crf = conf.get('use_crf', 0)
        self.conf = conf

    def _load_data(self):
        # for output tensor
        max_len = 0
        tokens_tensor = []
        att_mask_tensor = []
        tri_seq_label_tensor = []
        ent_seq_label_tensor = []
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

            if 'test' not in self.path:
                ent_seq_mention_mask_unit = []
                ent_seq_label = copy.copy(seq_label_template)
                for ent in info['ent_list']:
                    beg, mention = ent.split('@#@')[0], ent.split('@#@')[1]
                    beg = int(beg)
                    
                    if beg+len(mention)+1 >= self.max_seq_len:
                        offset_err += 1
                        continue
                    
                    tmp_mask = copy.copy(mask_template)
                    for i in range(beg+1, beg+len(mention)+1):
                        tmp_mask[i] = 1.0 / len(mention)
                        ent_seq_label[i] = 2
                    ent_seq_label[beg+1] = 1
                    ent_seq_mention_mask_unit.append(tmp_mask)

                ent_seq_mention_mask_unit = ent_seq_mention_mask_unit[:self.max_ent_len]
                for i in range(len(ent_seq_mention_mask_unit), self.max_ent_len):
                    ent_seq_mention_mask_unit.append(copy.copy(mask_template))
                
                tri_seq_mention_mask_unit = []
                tri_seq_label = copy.copy(seq_label_template)
                arg_role_label_unit = []

                for evt in info['event_list']:
                    evt_type = evt['event_type']
                    beg = evt['trigger_start_index']
                    mention = evt['trigger']
                    if beg+len(mention)+1 >= self.max_seq_len:
                        offset_err += 1
                        continue
                    tmp_mask = copy.copy(mask_template)
                    for i in range(beg+1, beg+len(mention)+1):
                        tmp_mask[i] = 1.0 / len(mention)
                        tri_seq_label[i] = 2
                    tri_seq_label[beg+1] = 1
                    tri_seq_mention_mask_unit.append(tmp_mask)

                    arg_role_map = {}
                    for a in evt['arguments']:
                        beg = a['argument_start_index']
                        mention = a['argument']
                        role = evt_type + '@#@' + a['role']
                        arg_role_map[str(beg) + '@#@' + mention] = role

                    tmp_role_label = []
                    for ent in info['ent_list']:
                        if ent in arg_role_map:
                            tmp_role_label.append(self.role_list.index(arg_role_map[ent]))
                        else:
                            tmp_role_label.append(0)
                    tmp_role_label = tmp_role_label[:self.max_ent_len]
                    for i in range(len(tmp_role_label), self.max_ent_len):
                        tmp_role_label.append(-100)
                    arg_role_label_unit.append(tmp_role_label)

                tri_seq_mention_mask_unit = tri_seq_mention_mask_unit[:self.max_evt_len]
                arg_role_label_unit = arg_role_label_unit[:self.max_evt_len]
                for i in range(len(tri_seq_mention_mask_unit), self.max_evt_len):
                    tri_seq_mention_mask_unit.append(copy.copy(mask_template))
                    arg_role_label_unit.append([-100] * self.max_ent_len)
                
                tri_seq_label_tensor.append(tri_seq_label)
                ent_seq_label_tensor.append(ent_seq_label)
                arg_role_label_tensor.append(arg_role_label_unit)
                tri_seq_mention_mask_tensor.append(tri_seq_mention_mask_unit)
                ent_seq_mention_mask_tensor.append(ent_seq_mention_mask_unit)
                # print(tokens)
                # print(tri_seq_label_tensor)
                # print(ent_seq_label_tensor)
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

        if offset_err:
            logging.warn("  越界: " + str(offset_err))
        if 'test' not in self.path:
            return tokens_tensor, att_mask_tensor, tri_seq_mention_mask_tensor, ent_seq_mention_mask_tensor, tri_seq_label_tensor, ent_seq_label_tensor, arg_role_label_tensor

        return tokens_tensor, att_mask_tensor
