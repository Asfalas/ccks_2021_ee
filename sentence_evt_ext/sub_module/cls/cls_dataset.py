import sys
sys.path.append('./')
import json
import logging
import copy
import torch

from transformers import BertTokenizer
from tqdm import tqdm
from sub_module.common.common_seq_tag_dataset import CommonDataSet
from sub_module.common.common_seq_tag_dataset import CommonSeqTagDataHandler

class ClsDataHandler(CommonSeqTagDataHandler):
    def __init__(self, path, conf, debug=0):
        super(ClsDataHandler, self).__init__(path, conf, debug=debug)
        self.path = path
        self.data = json.load(open(path))
        if debug:
            self.data = self.data[:200]
        self.event_schema = conf.get('event_schema', {})
        self.event_list = conf.get('event_list', [])
        self.max_seq_len = conf.get('max_seq_len', 400)
        self.max_ent_len = conf.get('max_ent_len', 32)
        self.max_role_len = conf.get('max_role_len', 7)
        self.evt_num = conf.get('evt_num', 34)
        self.add_cls = conf.get('add_cls', True)

    def _load_data(self):
        # for output tensor
        sent_token_id_tensor = []
        sent_att_mask_tensor = []
        evt_mention_mask_tensor = []
        evt_type_tensor = []
        arg_mention_mask_tensor = []
        arg_type_tensor = []
        arg_padding_num_tensor = []

        arg_err = 0
        evt_err = 0
        empty_arg_err = 0
        
        for event_info in tqdm(self.data):
            sentence = event_info['text']
            event = event_info['event']
            token_ids, att_mask = self.sentence_padding(sentence)
            event_type = event['event_type']

            # calc event offsets
            tmp_evt_offset = [event['trigger_start_index']+1, event['trigger_start_index']+len(event['trigger'])+1]
            valid = True
            for offset in tmp_evt_offset:
                if offset >= self.max_seq_len:
                    valid = False
                    evt_err += 1
                    break
            if not valid:
                continue

            assert sentence[tmp_evt_offset[0]-1 : tmp_evt_offset[1]-1] == event['trigger']

            # event mention mask
            tmp_evt_mask = [0.0] * self.max_seq_len
            for i in range(tmp_evt_offset[0], tmp_evt_offset[1]):
                tmp_evt_mask[i] = 1.0 / (tmp_evt_offset[1] - tmp_evt_offset[0])

            # arg mention mask & arg type
            arg_mention_mask_template = [0.0] * self.max_seq_len
            tmp_arg_masks = []
            tmp_arg_types = []
            tmp_arg_padding_mask = []

            valid = True
            if len(event.get('argument', [])) == 0:
                empty_arg_err += 1
#                 continue

            for arg in event.get('argument', []):
                offsets = [arg['argument_start_index'] +1, arg['argument_start_index'] + len(arg['argument']) + 1]

                valid = True
                for offset in offsets:
                    if offset >= self.max_seq_len:
                        valid = False
                        arg_err += 1
                        break
                if not valid:
                    continue

                tmp_offsets = copy.copy(arg_mention_mask_template)
                for i in range(offsets[0], offsets[1]):
                    tmp_offsets[i] = 1.0 / (offsets[1] - offsets[0])
                tmp_arg_masks.append(tmp_offsets)

                arg_type = arg['role']
                tmp_arg_types.append(self.event_schema[event_type].index(arg_type))
                # tmp_arg_padding_mask.append([1] * self.max_role_len)

            for i in range(len(tmp_arg_types), self.max_ent_len):
                tmp_arg_types += [-100]
                tmp_arg_masks += copy.copy([arg_mention_mask_template])
                # tmp_arg_padding_mask += [[0] * self.max_role_len]


            # put in output tensor
            sent_token_id_tensor.append(token_ids)
            sent_att_mask_tensor.append(att_mask)
            arg_type_tensor.append(tmp_arg_types)
            arg_padding_num_tensor.append(len(event['argument']))
            evt_mention_mask_tensor.append([copy.copy(tmp_evt_mask)])
            arg_mention_mask_tensor.append(tmp_arg_masks)
            evt_type_tensor.append(self.event_list.index(event_type))

        # transform to tensor
        sent_token_id_tensor = torch.LongTensor(sent_token_id_tensor)
        sent_att_mask_tensor = torch.LongTensor(sent_att_mask_tensor)
        arg_type_tensor = torch.LongTensor(arg_type_tensor)
        arg_mention_mask_tensor = torch.FloatTensor(arg_mention_mask_tensor)
        arg_padding_num_tensor = torch.FloatTensor(arg_padding_num_tensor)
        evt_type_tensor = torch.LongTensor(evt_type_tensor)
        evt_mention_mask_tensor = torch.FloatTensor(evt_mention_mask_tensor)

        if evt_err:
            logging.warn("  触发词越界: " + str(evt_err))
        if empty_arg_err:
            logging.info("  空论元列表: " + str(empty_arg_err))
        if arg_err:
            logging.info("  论元越界: " + str(arg_err))
        
        return sent_token_id_tensor, sent_att_mask_tensor, evt_mention_mask_tensor, arg_mention_mask_tensor, arg_padding_num_tensor, evt_type_tensor, arg_type_tensor

    def load(self):
        return self._load_data()
