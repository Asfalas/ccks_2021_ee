import sys
sys.path.append("./")
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class CommonSeqTagModel(nn.Module):
    def __init__(self, conf):
        super(CommonSeqTagModel, self).__init__()
        self.pretrained_model_name = conf.get("pretrained_model_name", 'bert-base-chinese')
        self.label = conf.get('label', ['O', 'B', 'I'])
        self.hidden_size = conf.get('hidden_size', 0)
        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.label)),
            nn.ReLU()
        )
        self.use_crf = conf.get('use_crf', 0)
        if self.use_crf:
            self.crf_layer = CRF(len(self.label), batch_first=True)

    def forward(self, inputs, labels):
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
        emissions = self.classifier(sequence_output)
        if self.use_crf:
            negloglike = -self.crf_layer(emissions, labels, mask=attention_mask)
            return emissions, negloglike
        else:
            return emissions

    def decode(self, emissions):
        if hasattr(self, "crf_layer"):
            return getattr(self, "crf_layer").decode(emissions)
        else:
            return None