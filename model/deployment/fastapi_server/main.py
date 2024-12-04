from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = FastAPI()


# Your model definition and tokenizer
class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits


model = CodeBERTClassifier()
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model.load_state_dict(torch.load('codebert_model.pth',map_location=torch.device('cpu')))
model.eval()
# Move model to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model.to(device)


# Define input data model
class CodeInput(BaseModel):
    code_samples: list


@app.post("/predict/")
def predict_code(code_input: CodeInput):
    code_samples = code_input.code_samples
    # Tokenization and prediction logic here
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

    # Move input tensors to the same device as the model
    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        outputs = model(tokens, attention_mask=masks)
        _, preds = torch.max(outputs, dim=1)
    prediction_labels = ["AI-generated" if pred == 1 else "Human-generated" for pred in preds.cpu().numpy()]
    return {"predictions": prediction_labels}