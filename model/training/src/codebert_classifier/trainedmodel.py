import torch
from transformers import RobertaTokenizer
import torch.nn as nn
from model import Autoencoder

# Assuming you already have the Autoencoder class definition from previous code

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
