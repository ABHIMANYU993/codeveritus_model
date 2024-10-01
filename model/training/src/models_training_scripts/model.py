import torch
from torch import nn
from safetensors.torch import load_file
from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from scratch_model import RobertaTokenizerFromScratch, RobertaForSequenceClassificationFromScratch

# Step 1: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(p=0.25)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 2: Load the trained model
model = CodeBERTClassifier()
# model.load_state_dict(
    # torch.load('E:/python-projects/llm/Trained_models/codebert_primary.pth'))

model.load_state_dict(
    load_file('E:/python-projects/models_training/Novathon-JSS/model.safetensors'))

model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Step 3: Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# Step 4: Preprocess the input code samples
def preprocess_input_code(code_samples):
    tokenized_samples = []
    attention_masks = []

    for code_sample in code_samples:
        tokenized_input = tokenizer(
            code_sample,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized_samples.append(tokenized_input['input_ids'].squeeze(0))
        attention_masks.append(tokenized_input['attention_mask'].squeeze(0))

    # Convert to PyTorch tensors
    tokens = torch.stack(tokenized_samples)
    masks = torch.stack(attention_masks)

    return tokens, masks


# Step 5: Make predictions
def predict_code_samples(model, code_samples):
    tokens, masks = preprocess_input_code(code_samples)

    # Move input tensors to the same device as the model
    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(tokens, attention_mask=masks)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    return probabilities.cpu().numpy()


code_samples = [""" hello world():
"""
]

# Make predictions
probabilities = predict_code_samples(model, code_samples)

# Print results with percentages for each code sample
for idx, (code, prob) in enumerate(zip(code_samples, probabilities)):
    ai_generated_prob = prob[1] * 100  # Percentage for AI-generated class (label 1)
    human_generated_prob = prob[0] * 100  # Percentage for Human-written class (label 0)
    print(f"Sample {idx + 1}:")
    print(f"Prediction: {ai_generated_prob:.2f}% AI-generated, {human_generated_prob:.2f}% Human-generated\n")

# adjust learning rate scheduler step

# add detailed comments to complex logic