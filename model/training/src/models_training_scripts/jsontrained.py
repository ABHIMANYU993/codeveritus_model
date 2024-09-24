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

# Step 1: Load your JSONL dataset
# Update this path to point to your own JSONL dataset
jsonl_file_path = "E:/python-projects/Datasets/AI-Human-Generated-Program-Code-Dataset-main/AI-Human-Generated-Program-Code-Dataset.jsonl"

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
    tokenized_samples = []
    attention_masks = []

    for code_sample in samples:
        # Tokenize and encode the code sample
        encoding = tokenizer(
            code_sample,
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate the text
            max_length=512,  # Max token length
            return_tensors='pt'  # Return PyTorch tensors
        )
        tokenized_samples.append(encoding['input_ids'].squeeze(0))  # Remove the batch dimension
        attention_masks.append(encoding['attention_mask'].squeeze(0))  # Attention masks for each sample

    return torch.stack(tokenized_samples), torch.stack(attention_masks), torch.tensor(labels)

tokens, attention_masks, labels = preprocess_dataset(code_samples, labels)

# Step 4: Split the dataset into train and validation
X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
    tokens, labels, attention_masks, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train, masks_train)
val_dataset = TensorDataset(X_val, y_val, masks_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 5: Define the model
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits

model = CodeBERTClassifier()

# Step 6: Optimizer, scheduler, and loss function
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
loss_fn = nn.CrossEntropyLoss()

scaler = GradScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 7: Training loop with mixed precision
def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10, patience=3):
    torch.cuda.empty_cache()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss, epochs_without_improvement = float('inf'), 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_predictions, total_predictions = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids, labels, attention_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            with autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, attention_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                with autocast(device_type='cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask)