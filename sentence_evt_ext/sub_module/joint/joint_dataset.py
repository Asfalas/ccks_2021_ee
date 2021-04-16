import sys
sys.path.append('./')
import json
import logging
import torch

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce
from sub_module.common.common_seq_tag_dataset import *

class JointDataHandler(CommonSeqTagDataHandler):
    def __init__(self, path, conf, debug=0):
        self.path = path
        self.data = [line for line in open(path).readlines()]
        logging.info('  debug 模式:' + ("True" if debug else "False"))
        if debug:
            self.data = self.data[:200]
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        
        event_schema = conf.get('event_schema', {})
        label1 = conf.get('schema', {}).get('label1', [])
        self.label1 = ['O']
        for i in label1:
            for a in event_schema[i]:
                if a=='None':
                    continue
                key = i + '@#@' + a if a != '时间' else '时间'
                self.label1 += ['B-' + key, 'I-' + key]

        label2 = conf.get('schema', {}).get('label2', [])
        self.label2 = ['O']
        for i in label2:
            for a in event_schema[i]:
                if a=='None':
                    continue
                key = i + '@#@' + a if a != '时间' else '时间'
                self.label2 += ['B-' + key, 'I-' + key]
                
        self.use_crf = conf.get('use_crf', 0)
        self.conf = conf

    def get_offset_list(self, info):
        arg_offsets_map = {}
        arg_offsets = []
        key_list = []
        for event_info in info['event_list']:
            et = event_info['event_type']
            for arg_info in event_info.get('arguments', []):
                beg = arg_info['argument_start_index']
                arg = arg_info['argument']
                end = beg + len(arg) - 1
                role = arg_info['role']
                key = et + '@#@' + role
                assert info['text'][beg: end+1] == arg
                offset_str = str(beg) + '_' + str(end)
                if offset_str not in arg_offsets_map:
                    arg_offsets_map[offset_str] = []
                arg_offsets_map[offset_str].append(key)
        for k, v in arg_offsets_map.items():
            key_list.append(v)
            beg, end = int(k.split('_')[0]), int(k.split('_')[1])
            arg_offsets.append([beg+1, end+1])
        return arg_offsets, key_list
        
    def _load_data(self):
        # for output tensor
        max_len = 0
        tokens_tensor = []
        att_mask_tensor = []
        label_tensor1 = []
        label_tensor2 = []

        offset_err = 0
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        for info in tqdm(self.data):
            info = json.loads(info)
            tokens = info['text']
            max_len = len(tokens) if len(tokens) > max_len else max_len
            token_ids, att_mask = self.sentence_padding(tokens)

            if 'test' not in self.path:
                offsets, key_list = self.get_offset_list(info)
                label_list1 = ['O' for i in range(self.max_seq_len)]
                label_list2 = ['O' for i in range(self.max_seq_len)]
                
                for offset, keys in zip(offsets, key_list):
                    if offset[1] >= self.max_seq_len:
                        valid = False
                        offset_err += 1
                        continue
                    for key in keys:
                        if '时间' in key:
                            key = '时间'
                        if 'B-' + key in self.label1:
                            label_list1[offset[0]] = 'B-' + key
                            for i in range(offset[0]+1, offset[1]+1):
                                label_list1[i] = 'I-' + key
                        
                        if 'B-' + key in self.label2:
                            label_list2[offset[0]] = 'B-' + key
                            for i in range(offset[0]+1, offset[1]+1):
                                label_list2[i] = 'I-' + key

                for i in range(len(label_list1)):
                    if token_ids[i] == pad_id:
                        if not self.use_crf:
                            label_list1[i] = -100
                        else:
                            label_list1[i] = 0
                    else:
                        label_list1[i] = self.label1.index(label_list1[i])
                
                for i in range(len(label_list2)):
                    if token_ids[i] == pad_id:
                        if not self.use_crf:
                            label_list2[i] = -100
                        else:
                            label_list2[i] = 0
                    else:
                        label_list2[i] = self.label2.index(label_list2[i])
                        
                label_tensor1.append(label_list1)
                label_tensor2.append(label_list2)

            tokens_tensor.append(token_ids)
            att_mask_tensor.append(att_mask)


        # transform to tensor
        tokens_tensor = torch.LongTensor(tokens_tensor)
        att_mask_tensor = torch.ByteTensor(att_mask_tensor)
        if 'test' not in self.path:
            label_tensor1 = torch.LongTensor(label_tensor1)
            label_tensor2 = torch.LongTensor(label_tensor2)
        else:
            label_tensor1 = [0] * tokens_tensor.size()[0]
            label_tensor2 = [0] * tokens_tensor.size()[0]

        if offset_err:
            logging.warn("  越界: " + str(offset_err))
        if max_len:
            logging.info("  最大句长: " + str(max_len))
        
        return tokens_tensor, att_mask_tensor, label_tensor1, label_tensor2
