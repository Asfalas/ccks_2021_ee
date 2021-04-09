import sys
sys.path.append("./")
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class JointModel(nn.Module):
    def __init__(self, conf):
        super(JointModel, self).__init__()
        self.pretrained_model_name = conf.get("pretrained_model_name", 'bert-base-chinese')
        label1 = conf.get('schema', {}).get('label1', [])
        self.label1 = ['O']
        for i in label1:
            self.label1 += ['B-' + i, 'I-' + i]

        label2 = conf.get('schema', {}).get('label2', [])
        self.label2 = ['O']
        for i in label2:
            self.label2 += ['B-' + i, 'I-' + i]
        
        self.hidden_size = conf.get('hidden_size', 0)
        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        self.classifier1 = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.label1)),
            nn.ReLU()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.label2)),
            nn.ReLU()
        )
        self.use_crf = conf.get('use_crf', 0)
        if self.use_crf:
            self.crf_layer1 = CRF(len(self.label1), batch_first=True)
            self.crf_layer2 = CRF(len(self.label2), batch_first=True)

    def forward(self, inputs, labels=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        token_type_ids:  (batch_size, max_seq_length)

        return: (batch_size, max_seq_length, num_labels)
        '''
        input_ids, attention_mask = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        emissions1 = self.classifier1(sequence_output)
        emissions2 = self.classifier2(sequence_output)
        if self.use_crf:
            negloglike1 = -self.crf_layer1(emissions1, labels[0], mask=attention_mask)
            negloglike2 = -self.crf_layer2(emissions2, labels[1], mask=attention_mask)
            negloglike = 0.5 * negloglike1 + 0.5 * negloglike2
            return (emissions1, emissions2), negloglike
        else:
            return (emissions1, emissions2)

    def decode(self, emissions):
        res = tuple()
        for emission, layer in zip(emissions, ['crf_layer1', 'crf_layer2']):
            if hasattr(self, layer):
                res += (getattr(self, layer).decode(emission), )
            else:
                res += (None, )
        return res