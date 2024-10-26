import os
import torch
from transformers import RobertaTokenizer
import torch.nn as nn
from model import Autoencoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define the path to the saved model
model_path = "C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/Saved_model/otherapproach.pth"

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load the trained model
autoencoder = Autoencoder(device)
autoencoder.load_state_dict(torch.load(model_path, weights_only=True))
autoencoder.eval()  # Set the model to evaluation mode

# Function to detect anomalies in new code samples
def detect_anomalies(code_sample):
    # Tokenize the new code sample
    tokenized = tokenizer.encode_plus(
        code_sample,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )['input_ids'].to(device)

    # Ensure tokenized input is float
    tokenized = tokenized.float()

    # Pass the tokenized input through the trained autoencoder
    with torch.no_grad():  # No need to track gradients during inference
        reconstructed = autoencoder(tokenized)
        loss_fn = nn.MSELoss()
        loss = loss_fn(reconstructed, tokenized)