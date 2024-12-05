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
