import sys
sys.path.append('./')
sys.path.append('../sentence_evt_ext/')
import json
import logging
import torch
import copy

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce
from sub_module.common.common_seq_tag_dataset import *

class ArtJointDataHandler(CommonSeqTagDataHandler):
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
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.role_list = conf.get("role_list", [])
        self.use_crf = conf.get('use_crf', 0)
        self.conf = conf

    def sentence_padding(self, tokens):
        if len(tokens) <= 512:
            token_ids = self.convert_tokens_to_ids(tokens)
        else:
            tokens = tokens[:self.max_seq_len]
            token_ids = []
            token_ids += self.convert_tokens_to_ids(tokens[:512])
            token_ids += self.convert_tokens_to_ids(tokens[512:])
        
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        att_mask = [1 for i in token_ids]
        length = len(token_ids)
        if length <= self.max_seq_len - 3:
            token_ids = token_ids + [pad_id] * (self.max_seq_len - 3 - length)
            att_mask = att_mask + [0] * (self.max_seq_len - 3 - length)
        else:
            token_ids = token_ids[:self.max_seq_len - 3]
            att_mask = att_mask[:self.max_seq_len - 3]
        
        token_ids_0 = [cls_id] + token_ids[:255] + [sep_id] + token_ids[255:510]
        att_mask_0 = [0] + att_mask[:255] + [0] + att_mask[255:510]

        token_ids_1 = [cls_id] + token_ids[255:510] + [sep_id] + token_ids[510:765]
        att_mask_1 = [0] + att_mask[255:510] + [0] + att_mask[510:765]
        
        assert len(token_ids_0) == len(att_mask_0) and len(token_ids_1) == len(att_mask_1)

        return token_ids_0, token_ids_1, att_mask_0, att_mask_1, token_ids

    def _load_data(self):
        # for output tensor
        max_len = 0
        front_tokens_tensor = []
        last_tokens_tensor = []

        front_att_mask_tensor = []
        last_att_mask_tensor = []

        tri_seq_label_tensor = []
        ent_seq_label_tensor = []
        tri_seq_mention_mask_tensor = []
        ent_seq_mention_mask_tensor = []
        arg_role_label_tensor = []

        offset_err = 0
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        mask_template = [0.0 for i in range(self.max_seq_len - 3)]

        for info in tqdm(self.data):
            if 'test' in self.path:
                info = json.loads(info)
            tokens = info['text']
            max_len = len(tokens) if len(tokens) > max_len else max_len
            token_ids_0, token_ids_1, att_mask_0, att_mask_1, token_ids = self.sentence_padding(tokens)

            seq_label_template = [0 for i in range(self.max_seq_len - 3)]
            for idx, ids in enumerate(token_ids):
                if ids == pad_id:
                    seq_label_template[idx] = -100

            front_tokens_tensor.append(token_ids_0)
            front_att_mask_tensor.append(att_mask_0)

            last_tokens_tensor.append(token_ids_1)
            last_att_mask_tensor.append(att_mask_1)

            if 'test' not in self.path:
                ent_seq_mention_mask_unit = []
                ent_seq_label = copy.copy(seq_label_template)
                for ent in info['ent_list']:
                    begs = ent['indexs']
                    mention = ent['ent']
                    
                    num = len(mention) * len(begs)
                    tmp_mask = copy.copy(mask_template)
                    for beg in begs:
                        if beg + len(mention) >= self.max_seq_len-3:
                            offset_err += 1
                            continue

                        for i in range(beg, beg + len(mention)):
                            tmp_mask[i] = 1.0 / num
                            ent_seq_label[i] = 2
                        ent_seq_label[beg] = 1
                    ent_seq_mention_mask_unit.append(tmp_mask)

                ent_seq_mention_mask_unit = ent_seq_mention_mask_unit[:self.max_ent_len]
                for i in range(len(ent_seq_mention_mask_unit), self.max_ent_len):
                    ent_seq_mention_mask_unit.append(copy.copy(mask_template))
                    
                tri_seq_mention_mask_unit = []
                tri_seq_label = copy.copy(seq_label_template)
                arg_role_label_unit = []

                for evt in info['event_list']:
                    evt_type = evt['event_type']
                    begs = evt['indexs']
                    mention = evt['trigger']
                    num = len(mention) * len(begs)
                    tmp_mask = copy.copy(mask_template)

                    for beg in begs:
                        if beg + len(mention) >= self.max_seq_len-3:
                            offset_err += 1
                            continue
                        for i in range(beg, beg + len(mention)):
                            tmp_mask[i] = 1.0 / num
                            tri_seq_label[i] = 2
                        tri_seq_label[beg] = 1
                    tri_seq_mention_mask_unit.append(tmp_mask)

                    arg_role_map = {}
                    for a in evt['arguments']:
                        begs = a['indexs']
                        mention = a['argument']
                        role = evt_type + '@#@' + a['role']
                        arg_role_map[str(begs) + '@#@' + mention] = role

                    tmp_role_label = []
                    for ent in info['ent_list']:
                        s = str(ent['indexs']) + ent['ent']
                        if s in arg_role_map:
                            tmp_role_label.append(self.role_list.index(arg_role_map[s]))
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
        front_tokens_tensor = torch.LongTensor(front_tokens_tensor)
        last_tokens_tensor = torch.LongTensor(last_tokens_tensor)
        front_att_mask_tensor = torch.ByteTensor(front_att_mask_tensor)
        last_att_mask_tensor = torch.ByteTensor(last_att_mask_tensor)
        if 'test' not in self.path:
           tri_seq_label_tensor = torch.LongTensor(tri_seq_label_tensor)
           ent_seq_label_tensor = torch.LongTensor(ent_seq_label_tensor)
           tri_seq_mention_mask_tensor = torch.FloatTensor(tri_seq_mention_mask_tensor)
           ent_seq_mention_mask_tensor = torch.FloatTensor(ent_seq_mention_mask_tensor)
           arg_role_label_tensor = torch.LongTensor(arg_role_label_tensor)

        if offset_err:
            logging.warn("  越界: " + str(offset_err))
        if 'test' not in self.path:
            return front_tokens_tensor, last_tokens_tensor, front_att_mask_tensor, last_att_mask_tensor, tri_seq_mention_mask_tensor, ent_seq_mention_mask_tensor, tri_seq_label_tensor, ent_seq_label_tensor, arg_role_label_tensor

        return front_tokens_tensor, last_tokens_tensor, front_att_mask_tensor, last_att_mask_tensor
