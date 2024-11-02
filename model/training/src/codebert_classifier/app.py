import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn as nn

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeBERTClassifier(nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        return logits


# Load the model with map_location set to 'cpu'
model = CodeBERTClassifier()
model.load_state_dict(
    torch.load('E:/python-projects/llm/Trained_models/codebert_model.pth', map_location=torch.device('cpu'))
)
model.eval()

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


# Define the request body schema
class CodeSample(BaseModel):
    code_sample: str


@app.post("/predict_code_sample")
async def predict_code_sample(code_sample: CodeSample):
    code = code_sample.code_sample
    # Tokenize input
    tokenized_input = tokenizer(
        code,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids = tokenized_input['input_ids'].to('cpu')
    attention_mask = tokenized_input['attention_mask'].to('cpu')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    prediction = 'AI-generated' if preds.item() == 1 else 'Human-generated'
    return {"prediction": prediction}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5000)

# increase batch size for faster training

# tweak dropout parameter for regularization

# reformat code to pep8 standards

# clean up exception handling block