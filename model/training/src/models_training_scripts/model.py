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