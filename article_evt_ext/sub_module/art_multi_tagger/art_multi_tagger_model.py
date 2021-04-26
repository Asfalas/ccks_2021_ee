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
       
        self.enum_classify_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.enum_list))
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

        enum_classify_input = torch.cat((outputs_0[:, 0, :], outputs_1[:, 0, :]), dim=-1)
        enum_logits = self.enum_classify_layer(enum_classify_input)

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
        
        return enum_logits, logits

    # def predict(self, inputs, use_gpu=True):
    #     with torch.no_grad():
    #         input_ids_0, input_ids_1, attention_mask_0, attention_mask_1 = inputs
    #         batch_size = input_ids_0.size()[0]
    #         outputs_0 = self.bert(
    #             input_ids=input_ids_0,
    #             attention_mask=attention_mask_0
    #         )[0]

    #         outputs_1 = self.bert(
    #             input_ids=input_ids_1,
    #             attention_mask=attention_mask_1
    #         )[0]

    #         if not self.use_tag_window:
    #             so1 = outputs_0[:, 1:511, :]
    #             so2 = outputs_1[:, 1:511, :]
    #             sequence_output = torch.cat((so1, so2), dim=1)
    #         else:
    #             so1 = outputs_0[:, :511, :]
    #             so2 = outputs_1[:, 1:, :]
    #             so = torch.cat((so1, so2), dim=1)
    #             s0 = so[:, :-2, :]
    #             s1 = so[:, 2:, :]
    #             s2 = so[:, 1:-1, :]
    #             sequence_output = 0.2 * s0 + 0.2 * s1 + 0.6 * s2
    #         if self.use_lstm:
    #             sequence_output, (_, _) = self.lstm_layer(sequence_output)
            
    #         tokens = torch.cat((input_ids_0[:, 1:511], input_ids_1[:, 1:511]), dim=-1)
            
    #         evt_logits = self.evt_tagger(sequence_output)
    #         ent_logits = self.ent_tagger(sequence_output)
            
    #         evt_label = torch.softmax(evt_logits, dim=-1)
    #         evt_label = torch.argmax(evt_label, dim=-1)
    #         evt_label = evt_label.squeeze().cpu().numpy()
            
    #         ent_label = torch.softmax(ent_logits, dim=-1)
    #         ent_label = torch.argmax(ent_label, dim=-1)
    #         ent_label = ent_label.squeeze().cpu().numpy()
            
            
    #         enum_classify_input = torch.cat((outputs_0[:, 0, :], outputs_1[:, 0, :]), dim=-1)
    #         enum_logits = self.enum_classify_layer(enum_classify_input)
            
    #         enum_label = torch.softmax(enum_logits, dim=-1)
    #         enum_label = torch.argmax(enum_logits, dim=-1)
    #         enum_label = enum_label.squeeze().cpu().numpy()

    #         result = []
    #         for i in range(batch_size):
    #             ent_seq_mention_mask_tensor = []
    #             evt_seq_mention_mask_tensor = []
    #             tmp_ents, tmp_evts = [], []
                
    #             self.calc_mention_mask(evt_label, evt_seq_mention_mask_tensor, i, tmp_evts, tokens)
    #             self.calc_mention_mask(ent_label, ent_seq_mention_mask_tensor, i, tmp_ents, tokens)
    #             ent_num = len(tmp_ents)
    #             evt_num = len(tmp_evts)
    #             if not ent_num or not evt_num:
    #                 result.append([])
    #                 continue
    #             ent_seq_mention_mask_tensor = torch.FloatTensor([ent_seq_mention_mask_tensor])
    #             evt_seq_mention_mask_tensor = torch.FloatTensor([evt_seq_mention_mask_tensor])
                
    #             if use_gpu:
    #                 ent_seq_mention_mask_tensor = ent_seq_mention_mask_tensor.cuda()
    #                 evt_seq_mention_mask_tensor = evt_seq_mention_mask_tensor.cuda()
                
    #             seq_embeds = sequence_output[i].unsqueeze(dim=0)

    #             evt_mention_embeds = torch.bmm(evt_seq_mention_mask_tensor, seq_embeds)
    #             ent_mention_embeds = torch.bmm(ent_seq_mention_mask_tensor, seq_embeds)
    #             merged_mention_embeds = torch.cat(
    #                 (
    #                     evt_mention_embeds.unsqueeze(dim=2).expand(-1, -1, ent_num, -1),
    #                     ent_mention_embeds.unsqueeze(dim=1).expand(-1, evt_num, -1, -1), 
    #                 ),
    #                 dim=-1).view(-1, ent_num * evt_num, self.tagger_embed_dim * 2)
    #             merged_mention_embeds = self.mention_merge_layer(merged_mention_embeds)
    #             role_logits = self.role_classification_layer(merged_mention_embeds).view(1, evt_num, ent_num, len(self.role_list))
    #             role_logits = torch.softmax(role_logits, dim=-1)
    #             role_logits = torch.argmax(role_logits, dim=-1).squeeze(dim=0).cpu().numpy()
                
    #             tmp_result = []
    #             for x in range(evt_num):
    #                 tmp = []
    #                 for y in range(ent_num):
    #                     tmp.append([tmp_evts[x], tmp_ents[y], role_logits[x][y]])
    #                 tmp_result.append(tmp)
    #             result.append(tmp_result)
                
    #         return result, enum_label
    
    # def calc_mention_mask(self, labels, masks, i, tmp_vector, tokens):
    #     j = 0
    #     mask_template = [0.0] * (self.real_max_seq_len)
    #     token_list = tokens[i]
    #     tokens_map = {}
    #     while j < self.real_max_seq_len:
    #         if labels[i][j] == 1:
    #             beg = j
    #             j += 1
    #             while j < self.real_max_seq_len and labels[i][j] == 2:
    #                 j += 1
    #             end = j
    #             mention = str(token_list[beg: end])
    #             if mention not in tokens_map:
    #                 tokens_map[mention] = []
    #             tokens_map[mention].append((beg, end))
    #         else:
    #             j += 1
    #     for mention, offs in tokens_map.items():
    #         tmp_mask = copy.copy(mask_template)
    #         for beg, end in offs:
    #             for x in range(beg, end):
    #                 tmp_mask[x] = 1.0 / (end - beg)
    #         masks.append(tmp_mask)
    #         tmp_vector.append([offs[0][0], offs[0][1]])