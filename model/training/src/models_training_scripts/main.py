import gdown
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Initialize FastAPI
app = FastAPI()

# Load CodeBERT Model
class CodeBERTClassifier(torch.nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits


def download_model_from_gdrive():
    url = 'https://drive.google.com/file/d/11_JwoS1l7Cim-NbRubhmVoE1QK3BYoy5/view?usp=drive_link'
    output = 'codebert_model.pth'