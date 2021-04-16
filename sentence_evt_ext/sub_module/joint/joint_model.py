import sys
sys.path.append("./")
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
import torch

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
        self.merged_embed_dim = conf.get("merged_embed_dim", 1024)
        self.role_list = conf.get("role_list", [])
        self.use_crf = conf.get('use_crf', 0)

        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        self.evt_tagger = nn.Sequential(
            nn.Linear(self.hidden_size, 3),
            nn.ReLU()
        )
        self.ent_tagger = nn.Sequential(
            nn.Linear(self.hidden_size, 3),
            nn.ReLU()
        )
        self.use_crf = conf.get('use_crf', 0)

        self.mention_merge_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.merged_embed_dim),
            nn.ReLU()
        )

        self.role_classification_layer = nn.Sequential(
                nn.Linear(self.merged_embed_dim, 2 * len(self.role_list)),
                nn.ReLU(),
                nn.Linear(2 * len(self.role_list), len(self.role_list))
            )
        # if self.use_crf:
        #     self.crf_layer1 = CRF(len(self.label1), batch_first=True)
        #     self.crf_layer2 = CRF(len(self.label2), batch_first=True)

    def forward(self, inputs):
        input_ids, attention_mask, tri_seq_mention_mask_tensor, ent_seq_mention_mask_tensor = inputs
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

        # batch_size * 30 * 768
        merged_mention_embeds = torch.cat(
            (
                event_mention_embeds.unsqueeze(dim=2).expand(-1, -1, self.max_ent_len, -1),
                ent_mention_embeds.unsqueeze(dim=1).expand(-1, self.max_evt_len, -1, -1), 
            ),
            dim=-1).view(-1, self.max_ent_len*self.max_evt_len, self.hidden_size * 2)
        
        role_logits = self.role_classification_layer(merged_mention_embeds)

        return (evt_logits, ent_logits, role_logits)


        # if self.use_crf:
        #     negloglike1 = -self.crf_layer1(emissions1, labels[0], mask=attention_mask)
        #     negloglike2 = -self.crf_layer2(emissions2, labels[1], mask=attention_mask)
        #     negloglike = 0.5 * negloglike1 + 0.5 * negloglike2
        #     return (emissions1, emissions2), negloglike
        # else:
        #     return (emissions1, emissions2)

    # def decode(self, emissions):
    #     res = tuple()
    #     for emission, layer in zip(emissions, ['crf_layer1', 'crf_layer2']):
    #         if hasattr(self, layer):
    #             res += (getattr(self, layer).decode(emission), )
    #         else:
    #             res += (None, )
    #     return res