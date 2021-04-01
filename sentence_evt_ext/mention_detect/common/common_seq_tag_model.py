import sys
sys.path.append("./")
import torch.nn as nn
from transformers import BertModel

class CommonSeqTagModel(nn.Module):
    def __init__(self, conf):
        super(CommonSeqTagModel, self).__init__()
        self.pretrained_model_name = conf.get("pretrained_model_name", 'bert-base-chinese')
        self.label = conf.get('label', ['O', 'B', 'I'])
        self.hidden_size = conf.get('hidden_size', 0)
        self.bert = BertModel.from_pretrained(self.pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, len(self.label)),
            nn.Relu()
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        '''
        input_ids:  (batch_size, max_seq_length)
        attention_mask:  (batch_size, max_seq_length)
        token_type_ids:  (batch_size, max_seq_length)

        return: (batch_size, max_seq_length, num_labels)
        '''
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        return logits
