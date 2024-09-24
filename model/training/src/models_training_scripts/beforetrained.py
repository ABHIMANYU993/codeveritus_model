import datasets
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# # Step 1: Load the datasets
time_complexity_dataset = load_dataset("codeparrot/codecomplex", cache_dir="E:/python-projects/Datasets", split="train")
# # ai_dataset1 = load_dataset("codeparrot/xlcost-text-to-code", "C++-program-level", cache_dir="E:/python-projects/Datasets", split="train")
# ai_dataset = load_dataset("codeparrot/xlcost-text-to-code", "Java-program-level", cache_dir="E:/python-projects/Datasets", split="train")
ai_dataset = load_dataset("codeparrot/codeparrot-clean",cache_dir="E:/python-projects/Datasets", split="train")
# # ai_dataset = datasets.concatenate_datasets([ai_dataset1, ai_dataset2, ai_dataset3])














print(f"Size of human dataset: {len(human_dataset)}")
print(f"Size of AI dataset: {len(ai_dataset)}")

# Sample sizes
human_sample_size = min(2000, len(human_dataset))
ai_sample_size = min(2000, len(ai_dataset))

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

human_tokens, human_labels, human_attention_masks = preprocess_dataset(human_dataset, 0, 'src')
ai_tokens, ai_labels, ai_attention_masks = preprocess_dataset(ai_dataset, 1, 'code')

tokens = human_tokens + ai_tokens
labels = human_labels + ai_labels
attention_masks = human_attention_masks + ai_attention_masks

tokens = torch.stack(tokens)
labels = torch.tensor(labels)
attention_masks = torch.stack(attention_masks)

# Step 3: Split the dataset
X_train, X_val, y_train, y_val, masks_train, masks_val = train_test_split(
    tokens, labels, attention_masks, test_size=0.2, random_state=42
)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train, masks_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val, masks_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 4: Define the CodeBERT model with increased dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)  # Increased dropout to 0.5

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits

model = CodeBERTClassifier()

# Step 5: Define optimizer, loss function, and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
loss_fn = nn.CrossEntropyLoss()

# Updated GradScaler and autocast for compatibility
scaler = GradScaler(device='cuda')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 6: Training loop with mixed-precision and early stopping
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