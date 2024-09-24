import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the datasets
human_dataset = load_dataset(data_dir="E:\python-projects\llm\.cache\huggingface\datasets\openai_humaneval\openai_humaneval", split="test")
ai_dataset = load_dataset(data_dir="E:\python-projects\llm\.cache\huggingface\datasets\codeparrot___codeparrot-clean", split="train")
# load_dataset(data_dir="/path/to/my_dataset")

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


# Preprocess datasets
human_tokens, human_labels, human_attention_masks = preprocess_dataset(human_dataset, 0, 'canonical_solution')
ai_tokens, ai_labels, ai_attention_masks = preprocess_dataset(ai_dataset, 1, 'content')

# Combine datasets
tokens = human_tokens + ai_tokens
labels = human_labels + ai_labels
attention_masks = human_attention_masks + ai_attention_masks

# Convert to PyTorch tensors
tokens = torch.stack(tokens)
labels = torch.tensor(labels)
attention_masks = torch.stack(attention_masks)

# Step 3: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(tokens, labels, attention_masks,
                                                                          test_size=0.2, random_state=42)

# Create PyTorch Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train, masks_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val, masks_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)


# Step 4: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Initialize model
model = CodeBERTClassifier()

# Step 5: Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)  # Weight decay for L2 regularization
loss_fn = nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


# Step 6: Training loop with early stopping
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=5, patience=3):
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
            optimizer.zero_grad()
            input_ids, labels, attention_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation loop