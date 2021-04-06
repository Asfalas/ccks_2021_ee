import sys
sys.path.append('./')
import json
import logging

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce
from sub_module.common.common_seq_tag_dataset import *

class ArgTagDataHandler(CommonSeqTagDataHandler):
    def get_offset_list(self, info):
        arg_offsets_set = set()
        arg_offsets = []
        for event_info in info['event_list']:
            for arg_info in event_info.get('arguments', []):
                beg = arg_info['argument_start_index']
                arg = arg_info['argument']
                end = beg + len(arg) - 1
                assert info['text'][beg: end+1] == arg
                offset_str = str(beg) + '_' + str(end)
                if offset_str in arg_offsets_set:
                    continue
                arg_offsets_set.add(offset_str)
                arg_offsets.append([beg+1, end+1])
        return arg_offsets