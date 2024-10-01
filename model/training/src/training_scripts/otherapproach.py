import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load the datasets
human_dataset = load_dataset("openai_humaneval", split="test")
ai_dataset = load_dataset("codeparrot/codeparrot-clean", split="train")

# Print dataset sizes
print(f"Size of human dataset: {len(human_dataset)}")
print(f"Size of AI dataset: {len(ai_dataset)}")

# Sample sizes for efficiency
human_sample_size = min(100, len(human_dataset))
ai_sample_size = min(100, len(ai_dataset))

# Select subsets
human_dataset = human_dataset.select(range(human_sample_size))
ai_dataset = ai_dataset.select(range(ai_sample_size))

# Step 2: Preprocess datasets
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

def preprocess_dataset(dataset, label, column_name):
    tokenized_samples, labels, attention_masks = [], [], []

    for code_sample in dataset[column_name]:
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

    return tokenized_samples, labels, attention_masks

# Preprocess human and AI datasets
human_tokens, human_labels, human_attention_masks = preprocess_dataset(human_dataset, 0, 'canonical_solution')
ai_tokens, ai_labels, ai_attention_masks = preprocess_dataset(ai_dataset, 1, 'content')

# Combine datasets
tokens = human_tokens + ai_tokens
labels = human_labels + ai_labels
attention_masks = human_attention_masks + ai_attention_masks

# Convert to PyTorch tensors
tokens = torch.stack(tokens).to(device)  # Move tokens to device immediately
labels = torch.tensor(labels).to(device)  # Move labels to device immediately
attention_masks = torch.stack(attention_masks).to(device)  # Move attention masks to device immediately

# Step 3: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
    tokens.cpu(), labels.cpu(), attention_masks.cpu(), test_size=0.2, random_state=42
)

# Create PyTorch Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train, masks_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val, masks_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

# Step 4: Define the CodeBERT model
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

# Initialize model
model = CodeBERTClassifier().to(device)  # Move model to device here

# Step 5: Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()  # Initialize gradient scaler for mixed precision

# Step 6: Training loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss, correct_predictions, total_predictions = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids, labels, attention_mask = (batch[0].to(device),
                                                 batch[1].to(device),
                                                 batch[2].to(device))

            with autocast():  # Use mixed precision
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()  # Scale the loss for backpropagation
            scaler.step(optimizer)  # Update the parameters
            scaler.update()  # Update the scaler

            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions.double() / total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation loop
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, attention_mask = (batch[0].to(device),
                                                     batch[1].to(device),
                                                     batch[2].to(device))
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                val_correct += torch.sum(preds == labels)
                val_total += labels.size(0)

        val_accuracy = val_correct.double() / val_total
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        model.train()

# Train the model
train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=5)

# Step 7: Save the model
torch.save(model.state_dict(), 'C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/Saved_model/codebert_model.pth')
print("Model saved to 'C:/Users/ABHIMANYU/PycharmProjects/cypherhackathon/Saved_model/codebert_model.pth'")

# add timing metrics to epoch loop

# tweak gradient accumulation steps

# add defensive checks for None values

# remove unused imports

# add debug logging to training loop

# add debug logging to training loop