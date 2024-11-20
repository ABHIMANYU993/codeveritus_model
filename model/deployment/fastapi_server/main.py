from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = FastAPI()


# Your model definition and tokenizer
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits

