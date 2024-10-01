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
            max_length=512,  # Adjust based on your needs
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tokens.append(encoded['input_ids'])
        labels.append(label)
    return tokens, labels


# Tokenize the dataset
tokens, labels = tokenize_data(dataset)
tokens = torch.cat(tokens)  # Concatenate token tensors
labels = np.array(labels)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(tokens, labels, test_size=0.2, random_state=42)


class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


# Instantiate the model
model = CodeBERTClassifier()

# Create DataLoader
train_data = TensorDataset(X_train, torch.tensor(y_train))
val_data = TensorDataset(X_val, torch.tensor(y_val))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):  # Adjust number of epochs
    for batch in train_loader:
        input_ids, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        _, predicted = torch.max(outputs.logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

accuracy = correct_predictions / total_predictions
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# tweak gradient accumulation steps

# comment out experimental code block

# refactor variable names for clarity