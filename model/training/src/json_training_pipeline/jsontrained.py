import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from safetensors.torch import save_file



# Step 1: Load your JSONL dataset
# Update this path to point to your own JSONL dataset
jsonl_file_path = "E:/python-projects/models_training/Novathon-JSS/converted_dataset.jsonl"

def load_jsonl_dataset(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# Load the dataset
dataset = load_jsonl_dataset(jsonl_file_path)

# Step 2: Extract ai_generated_code and human_generated_code, and assign labels
human_code_samples = [item['human_generated_code'] for item in dataset]
ai_code_samples = [item['ai_generated_code'] for item in dataset]

# Label human code as 0 and AI-generated code as 1
human_labels = [0] * len(human_code_samples)
ai_labels = [1] * len(ai_code_samples)

# Combine both datasets
code_samples = human_code_samples + ai_code_samples
labels = human_labels + ai_labels

# Step 3: Preprocess datasets with tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


def preprocess_dataset(samples, labels):
    encodings = tokenizer(
        samples,
        padding="max_length",  # Ensures all sequences are exactly 1024 tokens
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are tensor

    return input_ids, attention_masks, labels


tokens, attention_masks, labels = preprocess_dataset(code_samples, labels)

# Step 4: Split the dataset into train and validation
X_train, X_val, masks_train, masks_val, y_train, y_val = train_test_split(
    tokens, attention_masks, labels, test_size=0.2, random_state=42
)


train_dataset = TensorDataset(X_train, y_train, masks_train)
val_dataset = TensorDataset(X_val, y_val, masks_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)