---
title: Ai Code Detector
emoji: 🌖
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# Codeveritus Model Deployment - STREAMLIT_APP

## Deployment Environment Summary
This module defines the runtime configurations and application logic to deploy the Codeveritus classifier using the STREAMLIT SDK on Hugging Face Spaces. It contains the primary entrypoint `app.py` and configuration files such as `Dockerfile` and `requirements.txt` to run isolated predictions.

## Subsystem Configuration Details

### 1. Requirements File (`requirements.txt`)
Lists the Python dependencies required to load and execute the transformer-based model. It includes:
- `torch`: PyTorch CPU runtime library.
- `transformers`: Hugging Face transformers libraries for loading CodeBERT.
- `safetensors`: For loading lightweight model weights.
- `fastapi` and `uvicorn`: For serving APIs (or `streamlit` for Streamlit dashboards).

### 2. Dockerfile Configuration
Our Docker deployment utilizes Python images built on Alpine/Debian slim environments to minimize container size and security vulnerability surfaces:
```dockerfile
FROM python:3.9-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 3. Application Entrypoint (`app.py`)
The Python backend loads the `primary_model.safetensors` file into the CPU memory, initializes the tokenizer, compiles the input sequences, performs forward passes, and returns JSON-formatted prediction results:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from safetensors.torch import load_model
# ... model loading logic ...
```

## Local Execution Instructions
To test this deployment locally on your machine:
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the model weights `primary_model.safetensors` into this directory, renaming it to `model.safetensors` if necessary.
3. Run the server locally:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Query the API using a `curl` request:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"codeSnippet": "print(\"Hello World\")"}'
   ```

## Production Integration
In our final application stack, this deployment service runs as a stateless prediction engine. The Express backend relays code submissions to this microservice, keeping model weights separate from database operations.
