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