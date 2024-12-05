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
        torch.load('C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/codeberttraining/codebert_model.pth', map_location=torch.device('cpu')))
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

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits


def download_model_from_gdrive():
    url = 'https://drive.google.com/uc?export=download&id=1FeF4C07z0kJM-9t8ra80WtU37UiiUHCe'
    output = 'codebert_model.pth'
    gdown.download(url, output, quiet=False)


# Preprocess the code sample
def preprocess_input_code(code_samples, tokenizer):
    tokenized_samples = []
    attention_masks = []

    for code_sample in code_samples:
        tokenized_input = tokenizer(
            code_sample,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized_samples.append(tokenized_input['input_ids'].squeeze(0))
        attention_masks.append(tokenized_input['attention_mask'].squeeze(0))

    # Convert to PyTorch tensors
    tokens = torch.stack(tokenized_samples)
    masks = torch.stack(attention_masks)
    return tokens, masks


# # Make predictions
# def predict_code_sample(model, tokenizer, code_sample):
#     input_ids, attention_mask = preprocess_input_code(code_sample, tokenizer)
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#         _, pred = torch.max(outputs.logits, dim=1)
#     return "AI-generated" if pred.item() == 1 else "Human-generated"

# Step 5: Make predictions
def predict_code_samples(model, tokenizer, code_samples):
    tokens, masks = preprocess_input_code(code_samples, tokenizer)

    # Move input tensors to the same device as the model
    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(tokens, attention_mask=masks)
        _, preds = torch.max(outputs, dim=1)
        prediction_labels = ["AI-generated" if pred == 1 else "Human-generated" for pred in preds.cpu().numpy()]
    return prediction_labels


# Streamlit UI
st.title("AI Code Detector")
code_input = st.text_area("Enter your code sample here:", height=300)

if st.button("Predict"):
    model, tokenizer = load_model()
    prediction = predict_code_samples(model, tokenizer, code_input)
    st.write(f"Prediction: **{prediction}**")