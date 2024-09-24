import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Step 1: Load the datasets
human_dataset = load_dataset("openai_humaneval", split="test")
ai_dataset = load_dataset("codeparrot/codeparrot-clean", split="train")
# Print dataset sizes
print(f"Size of human dataset: {len(human_dataset)}")
print(f"Size of AI dataset: {len(ai_dataset)}")

# Sample sizes
human_sample_size = min(1000, len(human_dataset))
ai_sample_size = min(1000, len(ai_dataset))

# Select subsets
human_dataset = human_dataset.select(range(human_sample_size))
ai_dataset = ai_dataset.select(range(ai_sample_size))

# Step 2: Preprocess datasets
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


def preprocess_dataset(dataset, label, column_name):
    tokenized_samples = []
    labels = []
    attention_masks = []

    for idx, code_sample in enumerate(dataset[column_name]):
        tokenized_input = tokenizer(
            code_sample,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized_samples.append(tokenized_input['input_ids'].squeeze(0))
        labels.append(label)
        attention_masks.append(tokenized_input['attention_mask'].squeeze(0))

        # # If human-written code, add bug type if specified
        # if bug_type_column and label == 0:
        #     bug_types.append(dataset[bug_type_column][idx])
        # else:
        #     bug_types.append('ai_generated' if label == 1 else None)
    return tokenized_samples, labels, attention_masks
    # return tokenized_samples, labels, attention_masks, bug_types


# Preprocess human-written code dataset (using buggy_solution) and include bug_type
human_tokens, human_labels, human_attention_masks= preprocess_dataset(
    human_dataset,
    0,
    'canonical_solution'

)

# Preprocess AI-generated code dataset (using content)
ai_tokens, ai_labels, ai_attention_masks = preprocess_dataset(
    ai_dataset,
    1,
    'content'
)

# Combine datasets
tokens = human_tokens + ai_tokens
labels = human_labels + ai_labels
attention_masks = human_attention_masks + ai_attention_masks

# Convert to PyTorch tensors
tokens = torch.stack(tokens)
labels = torch.tensor(labels)
attention_masks = torch.stack(attention_masks)


# Step 3: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 4: Split the data into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(tokens))), test_size=0.2, random_state=42)

# Split the dataset
X_train, X_val = tokens[train_indices], tokens[val_indices]
y_train, y_val = labels[train_indices], labels[val_indices]
masks_train, masks_val = attention_masks[train_indices], attention_masks[val_indices]

# Create PyTorch Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train, masks_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val, masks_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=16)

# Step 5: Define training function with FP16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CodeBERTClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
scaler = GradScaler()

# Compute class weights to handle class imbalance
class_weights = torch.tensor([len(ai_labels) / len(human_labels), 1.0], device='cpu').to(
    'cuda')  # Modify to your dataset
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_epochs=6, patience=2):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids, labels, attention_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()

            with autocast():  # Mixed precision training
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            # Scale loss and perform backward pass
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation loop