import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Set Hugging Face cache directory to a writable location
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.makedirs(os.environ["HF_HOME"], exist_ok=True)  # Ensure the directory exists

# Initialize FastAPI
app = FastAPI()

# Load CodeBERT Model
class CodeBERTClassifier(torch.nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base",
            num_labels=2,
            cache_dir=os.environ["HF_HOME"]  # Use the custom cache directory
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

def load_model():
    model = CodeBERTClassifier().to('cpu')
    model.load_state_dict(torch.load('codebert_model.pth', map_location='cpu'), strict=False)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained(
        "microsoft/codebert-base",
        cache_dir=os.environ["HF_HOME"]  # Use the custom cache directory
    )
    return model, tokenizer

model, tokenizer = load_model()

# Request model
class CodeRequest(BaseModel):
    code_samples: list[str]

def preprocess_input_code(code_samples):
    inputs = tokenizer(code_samples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return inputs["input_ids"], inputs["attention_mask"]

# Predict function
def predict(code_samples):