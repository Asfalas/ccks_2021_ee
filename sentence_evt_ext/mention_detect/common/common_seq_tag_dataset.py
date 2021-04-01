import sys
sys.path.append('./')
import json
import logging
import torch

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce

class CommonSeqTagDataSet(Dataset):
    def __init__(self, dataset_name, data_handler, path, conf, debug=0):
        logging.info('  数据处理开始: ' + dataset_name + ' --> ' + path)
        dh = data_handler(path=path, conf=conf, debug=debug)
        self.data = dh.load()
        logging.info('  数据处理结束!\n')

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(x[idx] for x in self.data)

class CommonSeqTagDataHandler(object):
    def __init__(self, path, conf, debug=0):
        self.path = path
        self.data = [line for line in open(path).readlines()]
        logging.info('  debug 模式:' + ("True" if debug else "False"))
        if debug:
            self.data = self.data[:200]
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.label = conf.get('label', ['O', 'B', 'I'])
        self.conf = conf

    def convert_tokens_to_ids(self, tokens):
        t = [i for i in tokens][:self.max_seq_len]
        tokenized_text = self.tokenizer.encode_plus(t, add_special_tokens=False)
        input_ids = tokenized_text["input_ids"]
        return input_ids

    def sentence_padding(self, tokens):
        token_ids = self.convert_tokens_to_ids(tokens)
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        cls_id = self.tokenizer.convert_tokens_to_ids("[CLS]")
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        att_mask = [1.0 for i in token_ids]
        length = len(token_ids)
        if length <= self.max_seq_len - 2:
            token_ids = token_ids + [sep_id] + [pad_id] * (self.max_seq_len - 2 - length)
            att_mask = att_mask + [1.0] + [0.0] * (self.max_seq_len - 2 - length)
        else:
            token_ids = token_ids[:self.max_seq_len - 2] + [sep_id]
            att_mask = att_mask[:self.max_seq_len - 2] + [1.0]
        token_ids = [cls_id] + token_ids
        att_mask = [1.0] + att_mask
        assert len(token_ids) == len(att_mask)
        return token_ids, att_mask

    def get_offset_list(self, info):
        raise NotImplementedError()

    def _load_data(self):
        # for output tensor
        max_len = 0
        tokens_tensor = []
        att_mask_tensor = []
        label_tensor = []

        offset_err = 0
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        for info in tqdm(self.data):
            info = json.loads(info)
            tokens = info['text']
            max_len = len(tokens) if len(tokens) > max_len else max_len
            token_ids, att_mask = self.sentence_padding(tokens)

            offsets = self.get_offset_list(info)

            label_list = ['O' for i in range(self.max_seq_len)]
            for offset in offsets:
                if offset[1] >= self.max_seq_len:
                    valid = False
                    offset_err += 1
                    continue
                label_list[offset[0]] = 'B'
                for i in range(offset[0]+1, offset[1]+1):
                    label_list[i] = 'I'

            for i in range(len(label_list)):
                if token_ids[i] == pad_id:
                    label_list[i] = -100
                else:
                    label_list[i] = self.label.index(label_list[i])

            tokens_tensor.append(token_ids)
            att_mask_tensor.append(att_mask)
            label_tensor.append(label_list)


        # transform to tensor
        tokens_tensor = torch.LongTensor(tokens_tensor)
        att_mask_tensor = torch.LongTensor(att_mask_tensor)
        label_tensor = torch.LongTensor(label_tensor)

        if offset_err:
            logging.warn("  越界: " + str(offset_err))
        if max_len:
            logging.info("  最大句长: " + str(max_len))
        
        return tokens_tensor, att_mask_tensor, label_tensor

    def load(self):
        return self._load_data()
