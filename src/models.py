"""models.py: IndoBERT dengan keluaran 5 neuron (CE) atau 4 neuron (CORN). Dropout 0,3 dan pooler_output."""

import torch.nn as nn
from transformers import AutoModel

from src import config


class IndoBERTStandard(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        pooled = self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        return self.classifier(pooled)


class IndoBERTCORN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES - 1)

    def forward(self, input_ids, attention_mask):
        pooled = self.dropout(self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        return self.classifier(pooled)


def build_model(loss_type):
    if loss_type == "ce":
        return IndoBERTStandard().to(config.get_device())
    if loss_type == "corn":
        return IndoBERTCORN().to(config.get_device())
    raise ValueError(f"loss_type harus 'ce' atau 'corn', dapat: {loss_type}")
