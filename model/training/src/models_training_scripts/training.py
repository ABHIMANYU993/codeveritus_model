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