import os
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


# Step 1: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 2: Download the model from Google Drive if not present
def download_model_from_gdrive(url, destination):
    # Google Drive file ID extraction
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded the model file to {destination}")
    else:
        print("Failed to download the model file.")