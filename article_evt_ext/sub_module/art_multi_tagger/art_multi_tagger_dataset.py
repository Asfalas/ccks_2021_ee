import sys
sys.path.append('./')
sys.path.append('../sentence_evt_ext/')
import json
import logging
import torch
import copy
import random

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce
from sub_module.common.common_seq_tag_dataset import *

class ArtMultiTaggerDataHandler(CommonSeqTagDataHandler):
    def __init__(self, path, conf, debug=0):
        self.path = path
        if 'test' in path:
            self.data = [line for line in open(path, encoding='utf-8').readlines()]
        else:
            self.data = json.load(open(path, encoding='utf-8'))
        logging.info('  debug 模式:' + ("True" if debug else "False"))
        if debug:
            self.data = self.data[:200]
        self.max_seq_len = conf.get('max_seq_len', 768)
        self.real_max_seq_len = self.max_seq_len - 4
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.role_list = conf.get("role_list", [])
        self.label_map = conf.get("label_map", {})
        self.event_list = list(self.label_map.keys())
        self.use_crf = conf.get('use_crf', 0)
        self.enum_list = conf.get("enum_list", [])
        self.conf = conf

    def sentence_padding(self, tokens):
        if not tokens:
            tokens = ' '
        if len(tokens) <= 512:
            token_ids = self.convert_tokens_to_ids(tokens)
        else:
            tokens = tokens[:self.real_max_seq_len]
            token_ids = []
            token_ids += self.convert_tokens_to_ids(tokens[:512])
            token_ids += self.convert_tokens_to_ids(tokens[512:])

        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        att_mask = [1 for i in token_ids]
        length = len(token_ids)
        if length <= self.real_max_seq_len:
            token_ids = token_ids + [pad_id] * (self.real_max_seq_len - length)
            att_mask = att_mask + [0] * (self.real_max_seq_len - length)
        else:
            token_ids = token_ids[:self.real_max_seq_len]
            att_mask = att_mask[:self.real_max_seq_len]

        token_ids_0 = [cls_id] + token_ids[:510] + [sep_id]
        att_mask_0 = [0] + att_mask[:510] + [0]

        token_ids_1 = [cls_id] + token_ids[510:] + [sep_id]
        att_mask_1 = [0] + att_mask[510:] + [0]

        assert len(token_ids_0) == len(att_mask_0) and len(token_ids_1) == len(att_mask_1)

        return token_ids_0, token_ids_1, att_mask_0, att_mask_1, token_ids

    def _load_data(self):
        # for output tensor
        max_len = 0
        front_tokens_tensor = []
        last_tokens_tensor = []

        front_att_mask_tensor = []
        last_att_mask_tensor = []
        
        enum_type_tensor = []
        arg_seq_labels_tensor = []
        sample_flag_tensor = []

        offset_err = 0
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        for info in tqdm(self.data):
            if 'test' in self.path:
                info = json.loads(info)
            tokens = info['text']
            max_len = len(tokens) if len(tokens) > max_len else max_len
            token_ids_0, token_ids_1, att_mask_0, att_mask_1, token_ids = self.sentence_padding(tokens)
            
            tmp_seq_label = [0 for i in range(self.real_max_seq_len)]
            tmp_sample_flag = [0] * len(self.label_map)
            for idx, ids in enumerate(token_ids):
                if ids == pad_id:
                    seq_label_template[idx] = -100
            tmp_seq_label = tmp_seq_label * len(self.label_map)
            
            front_tokens_tensor.append(token_ids_0)
            front_att_mask_tensor.append(att_mask_0)

            last_tokens_tensor.append(token_ids_1)
            last_att_mask_tensor.append(att_mask_1)

            if 'test' not in self.path:
                enum = info['enum']
                enum_type_tensor.append(self.enum_list.index(enum))

                for evt in info['event_list']:
                    evt_type = evt['event_type']
                    event_id = self.event_list.index(evt_type)
                    if evt['arguments']:
                        tmp_sample_flag[event_id] = 1
                    arg_role_map = {}
                    for a in evt['arguments']:
                        begs = a['indexs']
                        mention = a['argument']
                        role_id = self.label_map[event_id].index(a['role'])
                        for beg in begs:
                            if beg + len(mention) > self.real_max_seq_len:
                                offset_err += 1
                                continue
                            for i in range(beg, beg + len(mention)):
                                tmp_seq_label[event_id][i] = role_id * 2 + 2
                            tmp_seq_label[event_id][beg] = role_id * 2 + 1
                
                # 负采样
                if 'train' in self.path:
                    for i in range(len(tmp_sample_flag)):
                        if tmp_sample_flag[i] == 0:
                            if random.random() < 0.125:
                                tmp_sample_flag = 1
                else:
                    tmp_sample_flag = [1] * len(self.label_map)
                
                arg_seq_labels_tensor.append(tmp_seq_label)
                sample_flag_tensor.append(tmp_sample_flag)

        # transform to tensor
        front_tokens_tensor = torch.LongTensor(front_tokens_tensor)
        last_tokens_tensor = torch.LongTensor(last_tokens_tensor)
        front_att_mask_tensor = torch.ByteTensor(front_att_mask_tensor)
        last_att_mask_tensor = torch.ByteTensor(last_att_mask_tensor)

        if 'test' not in self.path:
            enum_type_tensor = torch.LongTensor(enum_type_tensor)
            arg_seq_labels_tensor = torch.LongTensor(arg_seq_labels_tensor)
            sample_flag_tensor = torch.BoolTensor(sample_flag_tensor)

        if offset_err:
            logging.warn("  越界: " + str(offset_err))
        if 'test' not in self.path:
            return front_tokens_tensor, last_tokens_tensor, front_att_mask_tensor, last_att_mask_tensor, sample_flag_tensor, enum_type_tensor, arg_seq_labels_tensor

        return front_tokens_tensor, last_tokens_tensor, front_att_mask_tensor, last_att_mask_tensor
