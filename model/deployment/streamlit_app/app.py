import streamlit as st
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import gdown

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your model class
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits

# Function to download the model from Google Drive
def download_model_from_gdrive():
    url = 'https://drive.google.com/uc?export=download&id=1FeF4C07z0kJM-9t8ra80WtU37UiiUHCe'
    output = 'codebert_model.pth'
    gdown.download(url, output, quiet=False)