import sys
sys.path.append('./')
import math
import torch.nn as nn
import logging
import torch

from transformers import BertModel
from common.const import *

class ClsModel(nn.Module):
    def __init__(self, conf):
        super(ClsModel, self).__init__()
        self.embed_dim = conf.get('embed_dim', 768)
        # self.compressd_embed_dim = conf.get('compressd_embed_dim', 500)
        self.merged_embed_dim = conf.get('merged_embed_dim', 600)
        self.max_role_len = conf.get('max_role_len', 7)
        self.max_ent_len = conf.get('max_ent_len', 32)
        self.event_num = conf.get('evt_num', 34)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        self.use_gpu = conf.get('use_gpu', False)
        self.conf = conf

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
       
        self.mention_merge_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.merged_embed_dim),
            nn.ReLU()
        )
        # self.compress_embed_layer = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.compressd_embed_dim),
        #     nn.ReLU()
        # )
        self.arg_classification_layer = nn.Sequential(
                nn.Linear(self.merged_embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.max_role_len)
            )
        
        self.evt_classification_layer = nn.Sequential(
                nn.Linear(self.embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.event_num)
        )   


    def forward(self, inputs):
        sent_token_id_tensor, sent_att_mask_tensor, evt_mention_mask_tensor, arg_mention_mask_tensor, arg_padding_num_tensor = tuple(
            x for x in inputs)
        arg_padding_num_tensor = arg_padding_num_tensor.cpu().numpy()

        sent_embeds = self.bert(
            input_ids=sent_token_id_tensor,
            attention_mask=sent_att_mask_tensor
            )[0]
        
        event_mention_embeds = torch.bmm(evt_mention_mask_tensor, sent_embeds)
        arg_mention_embeds = torch.bmm(arg_mention_mask_tensor, sent_embeds)

        # batch * 1 * 200
        # classification
        merged_mention_embeds = torch.cat(
            (
                arg_mention_embeds, 
                event_mention_embeds.expand(-1, self.max_ent_len, -1)
            ),
            dim=-1)
        merged_mention_embeds = self.mention_merge_layer(merged_mention_embeds)

        evt_logits = self.evt_classification_layer(event_mention_embeds)
        arg_logits = self.arg_classification_layer(merged_mention_embeds)

        return evt_logits, arg_logits

