import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np


def load_code_files(folder_path, label):
    code_samples = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.c'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                code = file.read()
                code_samples.append((code, label))
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
    attention_masks = []
    for code, label in dataset:
        encoded = tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',