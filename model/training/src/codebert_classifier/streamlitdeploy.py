import streamlit as st
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import requests
import gdown


# Load the model from Google Drive or any other service
@st.cache_resource  # Cache the model, so it's not reloaded on every run
def load_model():
    # Download model file from Google Drive if needed
    # download_model_from_gdrive()

    # Load the model and tokenizer
    model = CodeBERTClassifier()
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model.load_state_dict(
        torch.load('C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/codeberttraining/codebert_model.pth', map_location=torch.device('cpu'),weights_only=True))
    model.eval()
    model.to(device)
    return model, tokenizer


device = torch.device('cpu')


# Your model definition and tokenizer
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)