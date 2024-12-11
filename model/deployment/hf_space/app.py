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


@app.get("/")
def home():
    return {"message": "API is running!"}


@app.post("/predict/")
async def predict_code(request: CodeRequest):
    probabilities = predict(request.code_samples)
    prediction_labels = []
    for prob in probabilities:
        ai_generated_prob = prob[1] * 100
        human_generated_prob = prob[0] * 100
        if ai_generated_prob > human_generated_prob:
            prediction_labels.append(f"{ai_generated_prob:.2f}% Of code similar to AI-generated code.")
        else:
            prediction_labels.append(f"{human_generated_prob:.2f}% Of code similar to Human-generated code.")
    return {"predictions": prediction_labels}


@app.post("/detect/")
async def detect_code(request: CodeRequest):
    probabilities = predict(request.code_samples)
    results = [{"AI": f"{prob[1]*100:.2f}%", "Human": f"{prob[0]*100:.2f}%"} for prob in probabilities]
    return {"predictions": results}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)