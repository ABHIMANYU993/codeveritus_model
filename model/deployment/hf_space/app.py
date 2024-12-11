import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from safetensors.torch import load_file
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Ensure CPU is always used
device = torch.device('cpu')

os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

app = FastAPI()

class CodeBERTClassifier(torch.nn.Module):
    def __init__(self):
        super(CodeBERTClassifier, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base",
            num_labels=2,
            cache_dir=os.environ["HF_HOME"]
        ).to(device)  # Ensure model is on CPU

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits


def load_model():
    model = CodeBERTClassifier()
    model.load_state_dict(load_file('model.safetensors'), strict=False)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained(
        "microsoft/codebert-base",
        cache_dir=os.environ["HF_HOME"]
    )
    return model, tokenizer

model, tokenizer = load_model()


class CodeRequest(BaseModel):
    code_samples: list[str]


def preprocess_input_code(code_samples):
    inputs = tokenizer(code_samples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)  # Move tensors to CPU


def predict(code_samples):
    tokens, masks = preprocess_input_code(code_samples)
    with torch.no_grad():
        logits = model(tokens, attention_mask=masks)
        probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()  # Keep on CPU for processing
    return probabilities
