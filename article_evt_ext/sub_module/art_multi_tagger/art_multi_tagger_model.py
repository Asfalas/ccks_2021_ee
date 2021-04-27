import sys
sys.path.append("./")
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
import torch
import copy
import logging

class ArtMultiTaggerModel(nn.Module):
    def __init__(self, conf):
        super(ArtMultiTaggerModel, self).__init__()
        self.pretrained_model_name = conf.get("pretrained_model_name", 'bert-base-chinese')
        self.hidden_size = conf.get('hidden_size', 0)
        self.max_seq_len = conf.get('max_seq_len', 512)
        self.real_max_seq_len = self.max_seq_len - 4
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.merged_embed_dim = conf.get("merged_embed_dim", 1024)
        self.lstm_hidden_dim = conf.get("lstm_hidden_size", 1024)
        self.use_lstm = conf.get("use_lstm", 0)
        self.role_list = conf.get("role_list", [])
        self.label_map = conf.get('label_map', {})
        self.event_list = list(self.label_map.keys())
        self.enum_list = conf.get("enum_list", [])
        self.use_crf = conf.get('use_crf', 0)
        logging.info("  use_crf: " + str(self.use_crf))
        self.use_tag_window = conf.get("use_tag_window", 0)
        logging.info("  use_tag_window: " + str(self.use_tag_window))
        
        self.tagger_embed_dim = self.lstm_hidden_dim if self.use_lstm else self.hidden_size

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        
        self.tagger_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.tagger_embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2 * len(self.label_map[self.event_list[i]]) + 1),
                nn.ReLU()
            )
            for i in range(len(self.event_list))
        )
       
        self.lstm_layer = nn.LSTM(input_size=self.hidden_size, hidden_size= self.lstm_hidden_dim // 2, num_layers=1,
                                     bidirectional=True,
                                     batch_first=True)
    
        # if self.use_crf:
        #     self.crf_layer1 = CRF(len(self.label1), batch_first=True)
        #     self.crf_layer2 = CRF(len(self.label2), batch_first=True)

    def forward(self, inputs):
        input_ids_0, input_ids_1, attention_mask_0, attention_mask_1 = inputs
        outputs_0 = self.bert(
            input_ids=input_ids_0,
            attention_mask=attention_mask_0
        )[0]

        outputs_1 = self.bert(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1
        )[0]

        if not self.use_tag_window:
            so1 = outputs_0[:, 1:511, :]
            so2 = outputs_1[:, 1:511, :]
            sequence_output = torch.cat((so1, so2), dim=1)
        else:
            so1 = outputs_0[:, :511, :]
            so2 = outputs_1[:, 1:, :]
            so = torch.cat((so1, so2), dim=1)
            s0 = so[:, :-2, :]
            s1 = so[:, 2:, :]
            s2 = so[:, 1:-1, :]
            sequence_output = 0.2 * s0 + 0.2 * s1 + 0.6 * s2
        
        if self.use_lstm:
            sequence_output, (_, _) = self.lstm_layer(sequence_output)
        
        logits = [tagger(sequence_output) for tagger in self.tagger_list]
        
        return logits

    def predict(self, inputs, use_gpu=True):
        with torch.no_grad():
            input_ids_0, input_ids_1, attention_mask_0, attention_mask_1 = inputs
            batch_size = input_ids_0.size()[0]
            outputs_0 = self.bert(
                input_ids=input_ids_0,
                attention_mask=attention_mask_0
            )[0]

            outputs_1 = self.bert(
                input_ids=input_ids_1,
                attention_mask=attention_mask_1
            )[0]

            if not self.use_tag_window:
                so1 = outputs_0[:, 1:511, :]
                so2 = outputs_1[:, 1:511, :]
                sequence_output = torch.cat((so1, so2), dim=1)
            else:
                so1 = outputs_0[:, :511, :]
                so2 = outputs_1[:, 1:, :]
                so = torch.cat((so1, so2), dim=1)
                s0 = so[:, :-2, :]
                s1 = so[:, 2:, :]
                s2 = so[:, 1:-1, :]
                sequence_output = 0.2 * s0 + 0.2 * s1 + 0.6 * s2
            if self.use_lstm:
                sequence_output, (_, _) = self.lstm_layer(sequence_output)
            
            logits = [tagger(sequence_output) for tagger in self.tagger_list]

            result = []
            for i in range(batch_size):
                logit = [l[i] for l in logits]
                res = self.calc_mention_mask(logit)
                result.append(res)
                
            return result
    
    def calc_mention_mask(self, logits):
        res = []
        for i in range(len(logits)):
            logit = logits[i]
            logit = torch.softmax(logit, dim=-1)
            logit = torch.argmax(logit, dim=-1)
            j = 0
            while j < self.real_max_seq_len:
                if logit[j] % 2 == 1:
                    beg = j
                    role_id = int(logit[j] / 2)
                    j += 1
                    while j < self.real_max_seq_len and logit[j] == 2 * role_id + 2:
                        j += 1
                    end = j
                    res.append((beg, end, i, role_id))
                else:
                    j += 1
        return res