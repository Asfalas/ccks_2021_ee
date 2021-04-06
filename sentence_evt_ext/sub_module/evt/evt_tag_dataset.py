import sys
sys.path.append('./')
import json
import logging

from transformers import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from _functools import reduce
from sub_module.common.common_seq_tag_dataset import *

class EvtTagDataHandler(CommonSeqTagDataHandler):
    def get_offset_list(self, info):
        evt_offsets = []
        for event_info in info['event_list']:
            beg = event_info['trigger_start_index']
            trg = event_info['trigger']
            end = beg + len(trg) - 1
            assert info['text'][beg: end+1] == trg
            evt_offsets.append([beg+1, end+1])
        return evt_offsets