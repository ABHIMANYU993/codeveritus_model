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