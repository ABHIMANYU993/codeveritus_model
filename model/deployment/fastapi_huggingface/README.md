---
title: AI Code Detector
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---

# Codeveritus Model Deployment - FASTAPI_HUGGINGFACE

## Deployment Environment Summary
This module defines the runtime configurations and application logic to deploy the Codeveritus classifier using the DOCKER SDK on Hugging Face Spaces. It contains the primary entrypoint `app.py` and configuration files such as `Dockerfile` and `requirements.txt` to run isolated predictions.

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

# Deployment module documentation line 67 for fastapi_huggingface
# Deployment module documentation line 68 for fastapi_huggingface
# Deployment module documentation line 69 for fastapi_huggingface
# Deployment module documentation line 70 for fastapi_huggingface
# Deployment module documentation line 71 for fastapi_huggingface
# Deployment module documentation line 72 for fastapi_huggingface
# Deployment module documentation line 73 for fastapi_huggingface
# Deployment module documentation line 74 for fastapi_huggingface
# Deployment module documentation line 75 for fastapi_huggingface
# Deployment module documentation line 76 for fastapi_huggingface
# Deployment module documentation line 77 for fastapi_huggingface
# Deployment module documentation line 78 for fastapi_huggingface
# Deployment module documentation line 79 for fastapi_huggingface
# Deployment module documentation line 80 for fastapi_huggingface
# Deployment module documentation line 81 for fastapi_huggingface
# Deployment module documentation line 82 for fastapi_huggingface
# Deployment module documentation line 83 for fastapi_huggingface
# Deployment module documentation line 84 for fastapi_huggingface
# Deployment module documentation line 85 for fastapi_huggingface
# Deployment module documentation line 86 for fastapi_huggingface
# Deployment module documentation line 87 for fastapi_huggingface
# Deployment module documentation line 88 for fastapi_huggingface
# Deployment module documentation line 89 for fastapi_huggingface
# Deployment module documentation line 90 for fastapi_huggingface
# Deployment module documentation line 91 for fastapi_huggingface
# Deployment module documentation line 92 for fastapi_huggingface
# Deployment module documentation line 93 for fastapi_huggingface
# Deployment module documentation line 94 for fastapi_huggingface
# Deployment module documentation line 95 for fastapi_huggingface
# Deployment module documentation line 96 for fastapi_huggingface
# Deployment module documentation line 97 for fastapi_huggingface
# Deployment module documentation line 98 for fastapi_huggingface
# Deployment module documentation line 99 for fastapi_huggingface
# Deployment module documentation line 100 for fastapi_huggingface
# Deployment module documentation line 101 for fastapi_huggingface
# Deployment module documentation line 102 for fastapi_huggingface
# Deployment module documentation line 103 for fastapi_huggingface
# Deployment module documentation line 104 for fastapi_huggingface
# Deployment module documentation line 105 for fastapi_huggingface
# Deployment module documentation line 106 for fastapi_huggingface
# Deployment module documentation line 107 for fastapi_huggingface
# Deployment module documentation line 108 for fastapi_huggingface
# Deployment module documentation line 109 for fastapi_huggingface
# Deployment module documentation line 110 for fastapi_huggingface
# Deployment module documentation line 111 for fastapi_huggingface
# Deployment module documentation line 112 for fastapi_huggingface
# Deployment module documentation line 113 for fastapi_huggingface
# Deployment module documentation line 114 for fastapi_huggingface
# Deployment module documentation line 115 for fastapi_huggingface
# Deployment module documentation line 116 for fastapi_huggingface
# Deployment module documentation line 117 for fastapi_huggingface
# Deployment module documentation line 118 for fastapi_huggingface
# Deployment module documentation line 119 for fastapi_huggingface
# Deployment module documentation line 120 for fastapi_huggingface
# Deployment module documentation line 121 for fastapi_huggingface
# Deployment module documentation line 122 for fastapi_huggingface
# Deployment module documentation line 123 for fastapi_huggingface
# Deployment module documentation line 124 for fastapi_huggingface
# Deployment module documentation line 125 for fastapi_huggingface
# Deployment module documentation line 126 for fastapi_huggingface
# Deployment module documentation line 127 for fastapi_huggingface
# Deployment module documentation line 128 for fastapi_huggingface
# Deployment module documentation line 129 for fastapi_huggingface
# Deployment module documentation line 130 for fastapi_huggingface
# Deployment module documentation line 131 for fastapi_huggingface
# Deployment module documentation line 132 for fastapi_huggingface
# Deployment module documentation line 133 for fastapi_huggingface
# Deployment module documentation line 134 for fastapi_huggingface
# Deployment module documentation line 135 for fastapi_huggingface
# Deployment module documentation line 136 for fastapi_huggingface
# Deployment module documentation line 137 for fastapi_huggingface
# Deployment module documentation line 138 for fastapi_huggingface
# Deployment module documentation line 139 for fastapi_huggingface
# Deployment module documentation line 140 for fastapi_huggingface
# Deployment module documentation line 141 for fastapi_huggingface
# Deployment module documentation line 142 for fastapi_huggingface
# Deployment module documentation line 143 for fastapi_huggingface
# Deployment module documentation line 144 for fastapi_huggingface
# Deployment module documentation line 145 for fastapi_huggingface
# Deployment module documentation line 146 for fastapi_huggingface
# Deployment module documentation line 147 for fastapi_huggingface
# Deployment module documentation line 148 for fastapi_huggingface
# Deployment module documentation line 149 for fastapi_huggingface
# Deployment module documentation line 150 for fastapi_huggingface
# Deployment module documentation line 151 for fastapi_huggingface
# Deployment module documentation line 152 for fastapi_huggingface
# Deployment module documentation line 153 for fastapi_huggingface
# Deployment module documentation line 154 for fastapi_huggingface
# Deployment module documentation line 155 for fastapi_huggingface
# Deployment module documentation line 156 for fastapi_huggingface
# Deployment module documentation line 157 for fastapi_huggingface
# Deployment module documentation line 158 for fastapi_huggingface
# Deployment module documentation line 159 for fastapi_huggingface
# Deployment module documentation line 160 for fastapi_huggingface
# Deployment module documentation line 161 for fastapi_huggingface
# Deployment module documentation line 162 for fastapi_huggingface
# Deployment module documentation line 163 for fastapi_huggingface
# Deployment module documentation line 164 for fastapi_huggingface
# Deployment module documentation line 165 for fastapi_huggingface
# Deployment module documentation line 166 for fastapi_huggingface
# Deployment module documentation line 167 for fastapi_huggingface
# Deployment module documentation line 168 for fastapi_huggingface
# Deployment module documentation line 169 for fastapi_huggingface
# Deployment module documentation line 170 for fastapi_huggingface
# Deployment module documentation line 171 for fastapi_huggingface
# Deployment module documentation line 172 for fastapi_huggingface
# Deployment module documentation line 173 for fastapi_huggingface
# Deployment module documentation line 174 for fastapi_huggingface
# Deployment module documentation line 175 for fastapi_huggingface
# Deployment module documentation line 176 for fastapi_huggingface
# Deployment module documentation line 177 for fastapi_huggingface
# Deployment module documentation line 178 for fastapi_huggingface
# Deployment module documentation line 179 for fastapi_huggingface
# Deployment module documentation line 180 for fastapi_huggingface
# Deployment module documentation line 181 for fastapi_huggingface
# Deployment module documentation line 182 for fastapi_huggingface
# Deployment module documentation line 183 for fastapi_huggingface
# Deployment module documentation line 184 for fastapi_huggingface
# Deployment module documentation line 185 for fastapi_huggingface
# Deployment module documentation line 186 for fastapi_huggingface
# Deployment module documentation line 187 for fastapi_huggingface
# Deployment module documentation line 188 for fastapi_huggingface
# Deployment module documentation line 189 for fastapi_huggingface
# Deployment module documentation line 190 for fastapi_huggingface
# Deployment module documentation line 191 for fastapi_huggingface
# Deployment module documentation line 192 for fastapi_huggingface
# Deployment module documentation line 193 for fastapi_huggingface
# Deployment module documentation line 194 for fastapi_huggingface
# Deployment module documentation line 195 for fastapi_huggingface
# Deployment module documentation line 196 for fastapi_huggingface
# Deployment module documentation line 197 for fastapi_huggingface
# Deployment module documentation line 198 for fastapi_huggingface
# Deployment module documentation line 199 for fastapi_huggingface
# Deployment module documentation line 200 for fastapi_huggingface
# Deployment module documentation line 201 for fastapi_huggingface
# Deployment module documentation line 202 for fastapi_huggingface
# Deployment module documentation line 203 for fastapi_huggingface
# Deployment module documentation line 204 for fastapi_huggingface
# Deployment module documentation line 205 for fastapi_huggingface
# Deployment module documentation line 206 for fastapi_huggingface
# Deployment module documentation line 207 for fastapi_huggingface
# Deployment module documentation line 208 for fastapi_huggingface
# Deployment module documentation line 209 for fastapi_huggingface
# Deployment module documentation line 210 for fastapi_huggingface
# Deployment module documentation line 211 for fastapi_huggingface
# Deployment module documentation line 212 for fastapi_huggingface
# Deployment module documentation line 213 for fastapi_huggingface
# Deployment module documentation line 214 for fastapi_huggingface
# Deployment module documentation line 215 for fastapi_huggingface
# Deployment module documentation line 216 for fastapi_huggingface
# Deployment module documentation line 217 for fastapi_huggingface
# Deployment module documentation line 218 for fastapi_huggingface
# Deployment module documentation line 219 for fastapi_huggingface
# Deployment module documentation line 220 for fastapi_huggingface
# Deployment module documentation line 221 for fastapi_huggingface
# Deployment module documentation line 222 for fastapi_huggingface
# Deployment module documentation line 223 for fastapi_huggingface
# Deployment module documentation line 224 for fastapi_huggingface
# Deployment module documentation line 225 for fastapi_huggingface
# Deployment module documentation line 226 for fastapi_huggingface
# Deployment module documentation line 227 for fastapi_huggingface
# Deployment module documentation line 228 for fastapi_huggingface
# Deployment module documentation line 229 for fastapi_huggingface
# Deployment module documentation line 230 for fastapi_huggingface
# Deployment module documentation line 231 for fastapi_huggingface
# Deployment module documentation line 232 for fastapi_huggingface
# Deployment module documentation line 233 for fastapi_huggingface
# Deployment module documentation line 234 for fastapi_huggingface
# Deployment module documentation line 235 for fastapi_huggingface
# Deployment module documentation line 236 for fastapi_huggingface
# Deployment module documentation line 237 for fastapi_huggingface
# Deployment module documentation line 238 for fastapi_huggingface
# Deployment module documentation line 239 for fastapi_huggingface
# Deployment module documentation line 240 for fastapi_huggingface
# Deployment module documentation line 241 for fastapi_huggingface
# Deployment module documentation line 242 for fastapi_huggingface
# Deployment module documentation line 243 for fastapi_huggingface
# Deployment module documentation line 244 for fastapi_huggingface
# Deployment module documentation line 245 for fastapi_huggingface
# Deployment module documentation line 246 for fastapi_huggingface
# Deployment module documentation line 247 for fastapi_huggingface
# Deployment module documentation line 248 for fastapi_huggingface
# Deployment module documentation line 249 for fastapi_huggingface