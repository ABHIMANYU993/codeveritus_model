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
