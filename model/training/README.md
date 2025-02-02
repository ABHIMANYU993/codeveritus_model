# Codeveritus Model Training & Research Pipeline

## Overview
This directory is dedicated to the core research, dataset token checking, training scripts, notebook validation, and SafeTensors model export routines for the AI vs Human Code Classifier.

## Model Selection Strategy
To classify code snippets with high fidelity, simple sequence classification models (like standard BERT or RoBERTa) are insufficient. They are trained primarily on natural language and lack an understanding of semantic constructs such as syntax trees, variables, class definitions, and keywords. We utilize `microsoft/codebert-base` as the base model because it was pre-trained on bimodal data (NL-PL pairs) spanning six programming languages (Python, Java, JavaScript, PHP, Ruby, and Go). CodeBERT allows us to encode source code snippets as dense vector representations that preserve syntax structure.

## Directory Contents
- `src/scratch_model/`: Early scripts using basic Multi-Layer Perceptrons (MLPs) and LSTM neural networks in PyTorch. This served as a baseline comparison before we moved to Transformer-based architectures.
- `src/training_scripts/`: Early CUDA configuration checks and baseline train loops using small local datasets.
- `src/models_training_scripts/`: Core production training scripts. Includes `model.py` (our classifier head wrapping CodeBERT), `training.py` (training utilities), and `main.py` (execution script). Includes specialized pipelines like `jsonltrain.py` and `jsontrain.py` to ingest different data formats.
- `src/token_checking/`: Vocabulary parsing and validation tools to analyze tokenizer behavior and manage Out-Of-Vocabulary (OOV) tokens.
- `src/json_training_pipeline/`: Data processing pipelines that format raw datasets and convert pickle-based model weights to Hugging Face SafeTensors format.
- `notebooks/experimental/`: Colab and local notebooks detailing early experiments, testing API parameters, and verifying Hugging Face integration.
- `notebooks/training/`: Training logs and fine-tuning parameters, including experimentations with gradient accumulation and K-Fold cross-validation.
- `notebooks/evaluation/`: Jupyter notebooks calculating accuracy, precision, recall, and F1-scores, and visualizing threshold charts to find the optimal decision boundary.

## Model Architecture Specifications
The model wraps the base CodeBERT encoder and appends a classification head:
```python
import torch.nn as nn
from transformers import AutoModel

class CodeBERTClassifier(nn.Module):
    def __init__(self, model_name='microsoft/codebert-base', num_labels=2, dropout_prob=0.1):
        super(CodeBERTClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use CLS token representation
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

## Detailed Training Methodology
To achieve maximum generalization and avoid overfitting on specific generative styles, we implement the following procedures:

### 1. Data Ingestion & Sequence Padding
Code snippets are processed through the CodeBERT tokenizer. Because code lengths vary significantly, we truncate inputs to a maximum sequence length of 512 tokens. Sequences shorter than 512 are padded using the `[PAD]` token, and attention masks are generated to prevent the encoder from paying attention to padded indexes.

### 2. Fine-Tuning Setup
- **Optimizer**: AdamW (Adam with Weight Decay) with a learning rate of `2e-5` and weight decay of `0.01`.
- **Learning Rate Scheduler**: Linear warmup scheduler over the first 10% of total training steps, followed by a linear decay to 0.
- **Loss Function**: Cross-Entropy Loss.
- **Batch Size**: 16 instances per device.

### 3. Gradient Accumulation
To overcome hardware memory constraints when working with the 512-token CodeBERT model on limited GPU memory, we implement gradient accumulation. Gradients are accumulated over 4 steps, effectively creating a batch size of 64:
```python
optimizer.zero_grad()
for step, batch in enumerate(train_dataloader):
    inputs, labels = batch
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 4. K-Fold Cross Validation
To ensure robust evaluation and prevent dataset split bias, K-Fold cross-validation ($K=5$) is executed. The dataset is partitioned into 5 stratified folds. For each fold:
- 4 folds are used for training and 1 fold for validation.
- Model performance is averaged across all 5 runs to get validation metrics.

## Evaluation Metrics
During testing, we analyze performance across several metrics:
1. **Accuracy**: Total correct predictions over total instances.
2. **Precision**: True Positives divided by (True Positives + False Positives). Ensures that human code is not falsely categorized as AI code.
3. **Recall**: True Positives divided by (True Positives + False Negatives). Ensures that AI code is correctly caught.
4. **F1-Score**: Harmonic mean of Precision and Recall. This serves as our primary metric to judge model quality:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

## Model Weights Serialization (SafeTensors)
We convert the traditional PyTorch `.bin` weights file into `.safetensors` format. SafeTensors offers significant advantages:
- **Safety**: Excludes python pickle files, which are prone to code injection vulnerabilities during deserialization.
- **Speed**: Enables zero-copy memory mapping, loading model weights up to 10x faster from disk.
The conversion is executed using our conversion script:
```python
from safetensors.torch import save_file
import torch

state_dict = torch.load('model.bin')
save_file(state_dict, 'primary_model.safetensors')
```

## Research References
1. Feng et al., "CodeBERT: A Pre-Trained Model for Programming Languages", arXiv:2009.08366.
2. PyTorch Optimization Guides for Transformer-based architectures.

## Hyperparameter Tuning Runs Logs (Summary)
Below is a historical trace of parameter configurations tested during the research phase:

| Run ID | Base Model | Batch Size | LR | Warmup Steps | Validation Accuracy | F1-Score |
|---|---|---|---|---|---|---|
| RUN-01 | codebert-base | 8 | 5e-5 | 100 | 89.2% | 0.887 |
| RUN-02 | codebert-base | 16 | 3e-5 | 200 | 92.4% | 0.921 |
| RUN-03 | codebert-base | 16 | 2e-5 | 500 | 94.8% | 0.946 |
| RUN-04 | codebert-base (Accum=4) | 64 | 2e-5 | 500 | 96.5% | 0.963 |
| RUN-05 | roberta-base | 16 | 2e-5 | 200 | 85.3% | 0.841 |

As seen in the runs, CodeBERT significantly outperformed RoBERTa-base, especially when paired with gradient accumulation to stabilize the updates.

## Usage Instructions for Training
To initiate a fine-tuning run locally on a system with a CUDA-supported GPU, execute the training script:
```bash
python3 src/models_training_scripts/main.py --data_path ./data/augmented_dataset.jsonl --epochs 5 --lr 2e-5 --output_dir ./trained_models
```
Monitor loss and accuracies outputted in real time. Once completed, convert the final checkpoint to safetensors.
