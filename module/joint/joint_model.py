import sys
sys.path.append("./")
import torch.nn as nn
from transformers import BertModel
# from torchcrf import CRF
import torch
import copy

class JointModel(nn.Module):
    def __init__(self, conf):
        super(JointModel, self).__init__()
        self.pretrained_model_name = conf.get("pretrained_model_name", 'bert-base-chinese')
        self.hidden_size = conf.get('hidden_size', 0)
        self.max_seq_len = conf.get('max_seq_len', 256)
        self.pretrained_model_name = conf.get('pretrained_model_name', 'bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        self.max_ent_len = conf.get("max_ent_len", 10)
        self.max_evt_len = conf.get("max_evt_len", 3)
        self.entity_list = conf.get("entity_list", [])
        self.event_list = conf.get("event_list", [])
        self.merged_embed_dim = conf.get("merged_embed_dim", 1024)
        self.role_list = conf.get("role_list", [])
        self.entity_embed_dim = conf.get("entity_embed_dim", 4)
        self.event_embed_dim = conf.get("event_embed_dim", 8)
        self.use_crf = conf.get('use_crf', 0)

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        self.evt_tagger = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * len(self.event_list) + 1),
            nn.ReLU()
        )
        self.ent_tagger = nn.Sequential(
            nn.Linear(self.hidden_size, 3),
            nn.ReLU()
        )
        
        self.evt_type_embedding_layer = nn.Embedding(len(self.event_list), embedding_dim=self.event_embed_dim, padding_idx=0)
        self.evt_type_embedding_layer.weight.requires_grad = True

        self.mention_merge_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.event_embed_dim, self.merged_embed_dim),
            # nn.Linear(self.hidden_size * 2, self.merged_embed_dim),
            nn.ReLU()
        )

        self.role_classification_layer = nn.Sequential(
                nn.Linear(self.merged_embed_dim, 2 * len(self.role_list)),
                nn.ReLU(),
                nn.Linear(2 * len(self.role_list), len(self.role_list))
            )
        

    def forward(self, inputs):
        input_ids, attention_mask, tri_seq_mention_mask_tensor, ent_seq_mention_mask_tensor, evt_type_label_tensor = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        evt_logits = self.evt_tagger(sequence_output)
        ent_logits = self.ent_tagger(sequence_output)

        # batch_size * 3 * max_seq_len, batch_size * max_seq_len * 768 --> batch_size * 3 * 768
        event_mention_embeds = torch.bmm(tri_seq_mention_mask_tensor, sequence_output)
        # batch_size * 10 * max_seq_len, batch_size * max_seq_len * 768 --> batch_size * 10 * 768
        ent_mention_embeds = torch.bmm(ent_seq_mention_mask_tensor, sequence_output)

        evt_type_embeds = self.evt_type_embedding_layer(evt_type_label_tensor)

        # batch_size * 30 * 768
        merged_mention_embeds = torch.cat(
            (
                event_mention_embeds.unsqueeze(dim=2).expand(-1, -1, self.max_ent_len, -1),
                ent_mention_embeds.unsqueeze(dim=1).expand(-1, self.max_evt_len, -1, -1),
                evt_type_embeds.unsqueeze(dim=2).expand(-1, -1, self.max_ent_len, -1)
            # ),dim=-1).view(-1, self.max_ent_len*self.max_evt_len, self.hidden_size * 2)
            ), dim=-1).view(-1, self.max_ent_len*self.max_evt_len, self.hidden_size * 2 + self.event_embed_dim)

        merged_mention_embeds = self.mention_merge_layer(merged_mention_embeds)
        role_logits = self.role_classification_layer(merged_mention_embeds)

        return (evt_logits, ent_logits, role_logits)

    def predict(self, inputs, use_gpu=True):
        with torch.no_grad():
            input_ids, attention_mask = inputs
            batch_size = input_ids.size()[0]
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            sequence_output = outputs[0]
            evt_logits = self.evt_tagger(sequence_output)
            ent_logits = self.ent_tagger(sequence_output)
            
            evt_label = torch.softmax(evt_logits, dim=-1)
            evt_label = torch.argmax(evt_label, dim=-1)
            evt_label = evt_label.squeeze().cpu().numpy()
            
            ent_label = torch.softmax(ent_logits, dim=-1)
            ent_label = torch.argmax(ent_label, dim=-1)
            ent_label = ent_label.squeeze().cpu().numpy()
            
            result = []
            for i in range(batch_size):
                ent_seq_mention_mask_tensor = []
                evt_seq_mention_mask_tensor = []
                tmp_ents, tmp_evts = [], []
                evt_types, ent_types = [], []
                
                self.calc_mention_mask(evt_label, evt_seq_mention_mask_tensor, i, tmp_evts, evt_types)
                self.calc_mention_mask(ent_label, ent_seq_mention_mask_tensor, i, tmp_ents, ent_types)
                ent_num = len(tmp_ents)
                evt_num = len(tmp_evts)
                if not ent_num or not evt_num:
                    result.append([])
                    continue
                ent_seq_mention_mask_tensor = torch.FloatTensor([ent_seq_mention_mask_tensor])
                evt_seq_mention_mask_tensor = torch.FloatTensor([evt_seq_mention_mask_tensor])
                evt_type_tensor = torch.LongTensor([evt_types])
                
                if use_gpu:
                    evt_seq_mention_mask_tensor = evt_seq_mention_mask_tensor.cuda()
                    ent_seq_mention_mask_tensor = ent_seq_mention_mask_tensor.cuda()
                    evt_type_tensor = evt_type_tensor.cuda()
                
                seq_embeds = sequence_output[i].unsqueeze(dim=0)

                evt_mention_embeds = torch.bmm(evt_seq_mention_mask_tensor, seq_embeds)
                ent_mention_embeds = torch.bmm(ent_seq_mention_mask_tensor, seq_embeds)
                evt_type_embeds = self.evt_type_embedding_layer(evt_type_tensor)

                merged_mention_embeds = torch.cat(
                    (
                        evt_mention_embeds.unsqueeze(dim=2).expand(-1, -1, ent_num, -1),
                        ent_mention_embeds.unsqueeze(dim=1).expand(-1, evt_num, -1, -1),
                        evt_type_embeds.unsqueeze(dim=2).expand(-1, -1, ent_num, -1)
                    ),
                    dim=-1).view(-1, ent_num * evt_num, self.hidden_size * 2 + self.event_embed_dim)
                merged_mention_embeds = self.mention_merge_layer(merged_mention_embeds)
                role_logits = self.role_classification_layer(merged_mention_embeds).view(1, evt_num, ent_num, len(self.role_list))
                role_logits = torch.softmax(role_logits, dim=-1)
                role_logits = torch.argmax(role_logits, dim=-1).squeeze(dim=0).cpu().numpy()

                tmp_result = []
                for x in range(evt_num):
                    tmp = []
                    for y in range(ent_num):
                        tmp.append([tmp_evts[x], tmp_ents[y], role_logits[x][y]])
                    tmp_result.append(tmp)
                result.append((tmp_result, tmp_ents, tmp_evts))
                
            return result
    
    def calc_mention_mask(self, labels, masks, i, tmp_vector, types):
        j = 0
        mask_template = [0.0] * self.max_seq_len
        while j < self.max_seq_len:
            if labels[i][j] % 2 == 1 and labels[i][j] != 0:
                tmpl = labels[i][j]
                beg = j
                j += 1
                while j < self.max_seq_len and labels[i][j] == tmpl+1:
                    j += 1
                end = j
                tmp_mask = copy.copy(mask_template)
                for x in range(beg, end):
                    tmp_mask[x] = 1.0 / (end - beg)
                types.append(int((tmpl)/2))
                masks.append(tmp_mask)
                tmp_vector.append((beg-1, end-1, int((tmpl)/2)))
            else:
                j += 1