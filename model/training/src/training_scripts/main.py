import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def load_code_files(folder_path, label):
    code_samples = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                code_samples.append((file.read(), label))
    return code_samples


# Load datasets
human_code_samples = load_code_files('C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/code_sample_public/2_SO-code/', 0)  # Label 0 for human code
ai_code_samples = load_code_files('C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/code_sample_public/3_GPT-code/', 1)  # Label 1 for AI code

# Combine datasets
dataset = human_code_samples + ai_code_samples

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


def tokenize_data(dataset):
    tokens = []
    labels = []
    for code, label in dataset:
        encoded = tokenizer.encode_plus(
            code,
            add_special_tokens=True,