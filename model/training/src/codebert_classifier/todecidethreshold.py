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

    # Return the reconstruction error as the anomaly score
    return loss.item()

# Function to load and get anomaly scores for all .c files in a folder
def get_anomaly_scores_from_folder(folder_path):
    anomaly_scores = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                code = file.read()
                # Get the anomaly score for each code file
                anomaly_score = detect_anomalies(code)
                anomaly_scores.append(anomaly_score)
                print(f"File: {filename}, Anomaly Score: {anomaly_score}")
    return anomaly_scores

# Define folder paths for human-written and AI-generated code
human_written_folder = "C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/code_sample_public/2_SO-code/"
ai_generated_folder = "C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/code_sample_public/3_GPT-code/"

# Get anomaly scores for human-written and AI-generated code
human_anomaly_scores = get_anomaly_scores_from_folder(human_written_folder)
ai_anomaly_scores = get_anomaly_scores_from_folder(ai_generated_folder)

# Combine the scores and generate labels (0 = human-written, 1 = AI-generated)
all_scores = human_anomaly_scores + ai_anomaly_scores
labels = [0] * len(human_anomaly_scores) + [1] * len(ai_anomaly_scores)

# Convert to numpy arrays for compatibility with scikit-learn
import numpy as np
all_scores = np.array(all_scores)
labels = np.array(labels)

from sklearn.metrics import precision_recall_curve, f1_score

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(labels, all_scores)

valid_mask = (precision + recall) > 0  # Avoid division by zero
f1_scores = np.zeros_like(precision)  # Initialize with zeros