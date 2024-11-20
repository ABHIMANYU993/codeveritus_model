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
