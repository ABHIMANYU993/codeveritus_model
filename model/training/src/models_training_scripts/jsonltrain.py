import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Step 1: Define the Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append({
                    "code": data["ai_generated_code","human_generated_code"],
                    "label": 0
                })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokenized = self.tokenizer(
            item["code"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

# Step 2: Load the Dataset
file_path = "converted_dataset.jsonl"  # Path to the converted dataset
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
dataset = CustomDataset(file_path, tokenizer)

# Step 3: Split the Dataset
train_size = int(0.7 * len(dataset))  # 70% training, 30% validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 4: Define the CodeBERT Model
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits

model = CodeBERTClassifier()

# Step 5: Define Optimizer, Loss Function, and Scheduler
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
loss_fn = nn.CrossEntropyLoss()

scaler = GradScaler()  # For mixed-precision training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 6: Training Loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10, patience=4):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss, epochs_without_improvement = float('inf'), 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss, correct_predictions, total_predictions = 0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Logging
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies

train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10
)


# Save the trained model
save_path = f"E:/python-projects/llm/Trained_models/ai-human-jsontrained{max(train_accuracies)}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Step 7: Plot Results
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# refactor variable names for clarity

# tweak dropout parameter for regularization

# add type hints to function signatures

# reformat code to pep8 standards

# improve error message formatting

# update tensorboard logging interval