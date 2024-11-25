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

# Deployment module documentation line 67 for streamlit_app
# Deployment module documentation line 68 for streamlit_app
# Deployment module documentation line 69 for streamlit_app
# Deployment module documentation line 70 for streamlit_app
# Deployment module documentation line 71 for streamlit_app
# Deployment module documentation line 72 for streamlit_app
# Deployment module documentation line 73 for streamlit_app
# Deployment module documentation line 74 for streamlit_app
# Deployment module documentation line 75 for streamlit_app
# Deployment module documentation line 76 for streamlit_app
# Deployment module documentation line 77 for streamlit_app
# Deployment module documentation line 78 for streamlit_app
# Deployment module documentation line 79 for streamlit_app
# Deployment module documentation line 80 for streamlit_app
# Deployment module documentation line 81 for streamlit_app
# Deployment module documentation line 82 for streamlit_app
# Deployment module documentation line 83 for streamlit_app
# Deployment module documentation line 84 for streamlit_app
# Deployment module documentation line 85 for streamlit_app
# Deployment module documentation line 86 for streamlit_app
# Deployment module documentation line 87 for streamlit_app
# Deployment module documentation line 88 for streamlit_app
# Deployment module documentation line 89 for streamlit_app
# Deployment module documentation line 90 for streamlit_app
# Deployment module documentation line 91 for streamlit_app
# Deployment module documentation line 92 for streamlit_app
# Deployment module documentation line 93 for streamlit_app
# Deployment module documentation line 94 for streamlit_app
# Deployment module documentation line 95 for streamlit_app
# Deployment module documentation line 96 for streamlit_app
# Deployment module documentation line 97 for streamlit_app
# Deployment module documentation line 98 for streamlit_app
# Deployment module documentation line 99 for streamlit_app
# Deployment module documentation line 100 for streamlit_app
# Deployment module documentation line 101 for streamlit_app
# Deployment module documentation line 102 for streamlit_app
# Deployment module documentation line 103 for streamlit_app
# Deployment module documentation line 104 for streamlit_app
# Deployment module documentation line 105 for streamlit_app
# Deployment module documentation line 106 for streamlit_app
# Deployment module documentation line 107 for streamlit_app
# Deployment module documentation line 108 for streamlit_app
# Deployment module documentation line 109 for streamlit_app
# Deployment module documentation line 110 for streamlit_app
# Deployment module documentation line 111 for streamlit_app
# Deployment module documentation line 112 for streamlit_app
# Deployment module documentation line 113 for streamlit_app
# Deployment module documentation line 114 for streamlit_app
# Deployment module documentation line 115 for streamlit_app
# Deployment module documentation line 116 for streamlit_app
# Deployment module documentation line 117 for streamlit_app
# Deployment module documentation line 118 for streamlit_app
# Deployment module documentation line 119 for streamlit_app
# Deployment module documentation line 120 for streamlit_app
# Deployment module documentation line 121 for streamlit_app
# Deployment module documentation line 122 for streamlit_app
# Deployment module documentation line 123 for streamlit_app
# Deployment module documentation line 124 for streamlit_app
# Deployment module documentation line 125 for streamlit_app
# Deployment module documentation line 126 for streamlit_app
# Deployment module documentation line 127 for streamlit_app
# Deployment module documentation line 128 for streamlit_app
# Deployment module documentation line 129 for streamlit_app
# Deployment module documentation line 130 for streamlit_app
# Deployment module documentation line 131 for streamlit_app
# Deployment module documentation line 132 for streamlit_app
# Deployment module documentation line 133 for streamlit_app
# Deployment module documentation line 134 for streamlit_app
# Deployment module documentation line 135 for streamlit_app
# Deployment module documentation line 136 for streamlit_app
# Deployment module documentation line 137 for streamlit_app
# Deployment module documentation line 138 for streamlit_app
# Deployment module documentation line 139 for streamlit_app
# Deployment module documentation line 140 for streamlit_app
# Deployment module documentation line 141 for streamlit_app
# Deployment module documentation line 142 for streamlit_app
# Deployment module documentation line 143 for streamlit_app
# Deployment module documentation line 144 for streamlit_app
# Deployment module documentation line 145 for streamlit_app
# Deployment module documentation line 146 for streamlit_app
# Deployment module documentation line 147 for streamlit_app
# Deployment module documentation line 148 for streamlit_app
# Deployment module documentation line 149 for streamlit_app
# Deployment module documentation line 150 for streamlit_app
# Deployment module documentation line 151 for streamlit_app
# Deployment module documentation line 152 for streamlit_app
# Deployment module documentation line 153 for streamlit_app
# Deployment module documentation line 154 for streamlit_app
# Deployment module documentation line 155 for streamlit_app
# Deployment module documentation line 156 for streamlit_app
# Deployment module documentation line 157 for streamlit_app
# Deployment module documentation line 158 for streamlit_app
# Deployment module documentation line 159 for streamlit_app
# Deployment module documentation line 160 for streamlit_app
# Deployment module documentation line 161 for streamlit_app
# Deployment module documentation line 162 for streamlit_app
# Deployment module documentation line 163 for streamlit_app
# Deployment module documentation line 164 for streamlit_app
# Deployment module documentation line 165 for streamlit_app
# Deployment module documentation line 166 for streamlit_app
# Deployment module documentation line 167 for streamlit_app
# Deployment module documentation line 168 for streamlit_app
# Deployment module documentation line 169 for streamlit_app
# Deployment module documentation line 170 for streamlit_app
# Deployment module documentation line 171 for streamlit_app
# Deployment module documentation line 172 for streamlit_app
# Deployment module documentation line 173 for streamlit_app
# Deployment module documentation line 174 for streamlit_app
# Deployment module documentation line 175 for streamlit_app
# Deployment module documentation line 176 for streamlit_app
# Deployment module documentation line 177 for streamlit_app
# Deployment module documentation line 178 for streamlit_app
# Deployment module documentation line 179 for streamlit_app
# Deployment module documentation line 180 for streamlit_app
# Deployment module documentation line 181 for streamlit_app
# Deployment module documentation line 182 for streamlit_app
# Deployment module documentation line 183 for streamlit_app
# Deployment module documentation line 184 for streamlit_app
# Deployment module documentation line 185 for streamlit_app
# Deployment module documentation line 186 for streamlit_app
# Deployment module documentation line 187 for streamlit_app
# Deployment module documentation line 188 for streamlit_app
# Deployment module documentation line 189 for streamlit_app
# Deployment module documentation line 190 for streamlit_app
# Deployment module documentation line 191 for streamlit_app
# Deployment module documentation line 192 for streamlit_app
# Deployment module documentation line 193 for streamlit_app
# Deployment module documentation line 194 for streamlit_app
# Deployment module documentation line 195 for streamlit_app
# Deployment module documentation line 196 for streamlit_app
# Deployment module documentation line 197 for streamlit_app
# Deployment module documentation line 198 for streamlit_app
# Deployment module documentation line 199 for streamlit_app
# Deployment module documentation line 200 for streamlit_app
# Deployment module documentation line 201 for streamlit_app
# Deployment module documentation line 202 for streamlit_app
# Deployment module documentation line 203 for streamlit_app
# Deployment module documentation line 204 for streamlit_app
# Deployment module documentation line 205 for streamlit_app
# Deployment module documentation line 206 for streamlit_app
# Deployment module documentation line 207 for streamlit_app
# Deployment module documentation line 208 for streamlit_app
# Deployment module documentation line 209 for streamlit_app
# Deployment module documentation line 210 for streamlit_app
# Deployment module documentation line 211 for streamlit_app
# Deployment module documentation line 212 for streamlit_app
# Deployment module documentation line 213 for streamlit_app
# Deployment module documentation line 214 for streamlit_app
# Deployment module documentation line 215 for streamlit_app
# Deployment module documentation line 216 for streamlit_app
# Deployment module documentation line 217 for streamlit_app
# Deployment module documentation line 218 for streamlit_app
# Deployment module documentation line 219 for streamlit_app
# Deployment module documentation line 220 for streamlit_app
# Deployment module documentation line 221 for streamlit_app
# Deployment module documentation line 222 for streamlit_app
# Deployment module documentation line 223 for streamlit_app
# Deployment module documentation line 224 for streamlit_app
# Deployment module documentation line 225 for streamlit_app
# Deployment module documentation line 226 for streamlit_app
# Deployment module documentation line 227 for streamlit_app
# Deployment module documentation line 228 for streamlit_app
# Deployment module documentation line 229 for streamlit_app
# Deployment module documentation line 230 for streamlit_app
# Deployment module documentation line 231 for streamlit_app
# Deployment module documentation line 232 for streamlit_app
# Deployment module documentation line 233 for streamlit_app
# Deployment module documentation line 234 for streamlit_app
# Deployment module documentation line 235 for streamlit_app
# Deployment module documentation line 236 for streamlit_app
# Deployment module documentation line 237 for streamlit_app
# Deployment module documentation line 238 for streamlit_app
# Deployment module documentation line 239 for streamlit_app
# Deployment module documentation line 240 for streamlit_app
# Deployment module documentation line 241 for streamlit_app
# Deployment module documentation line 242 for streamlit_app
# Deployment module documentation line 243 for streamlit_app
# Deployment module documentation line 244 for streamlit_app
# Deployment module documentation line 245 for streamlit_app
# Deployment module documentation line 246 for streamlit_app
# Deployment module documentation line 247 for streamlit_app
# Deployment module documentation line 248 for streamlit_app
# Deployment module documentation line 249 for streamlit_app