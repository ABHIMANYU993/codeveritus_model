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
            truncation=True,
            return_tensors='pt'
        )
        tokens.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels.append(label)
    return tokens, attention_masks, labels


# Tokenize the dataset
tokens, attention_masks, labels = tokenize_data(dataset)
tokens = torch.cat(tokens)
attention_masks = torch.cat(attention_masks)
labels = np.array(labels)

# Split dataset
X_train, X_val, y_train, y_val, train_masks, val_masks = train_test_split(
    tokens, labels, attention_masks, test_size=0.2, random_state=42
)

class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


# Instantiate the model
model = CodeBERTClassifier()

# Create DataLoader
train_data = TensorDataset(X_train, train_masks, torch.tensor(y_train))
val_data = TensorDataset(X_val, val_masks, torch.tensor(y_val))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# Define optimizer using PyTorch's AdamW implementation
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
num_epochs = 5  # Increase epochs if needed
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

accuracy = correct_predictions / total_predictions
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# tweak gradient accumulation steps

# comment out experimental code block

# update batch normalization momentum

# increase batch size for faster training