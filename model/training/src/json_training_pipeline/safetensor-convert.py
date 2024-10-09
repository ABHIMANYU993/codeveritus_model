from safetensors.torch import save_file

# Get model's state_dict

import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
from safetensors.torch import load_file


# Step 1: Define the CodeBERT model with Dropout
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.25)  # Add a dropout layer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)  # Apply dropout
        return logits


# Step 2: Load the trained model
model = CodeBERTClassifier()
# model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_codeparrot-v20.8955223880597015.pth'))
# model.load_state_dict(
# torch.load('E:/python-projects/llm/Trained_models/codebert_codeparrotjsontrained0.9886363636363636.pth'))
model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_primary.pth'))

# model.load_state_dict(
#     load_file('E:/python-projects/models_training/Novathon-JSS/codebert_json10-0.9587301587301588.safetensors'),
#     strict=False)

# model.load_state_dict(torch.load('E:/python-projects/llm/Trained_models/codebert_codeparrotjsontrained1.0.pth'))
state_dict = model.state_dict()
# Define save path
save_path = "model.safetensors"

# Save in .safetensors format
save_file(state_dict, save_path)
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Step 3: Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")