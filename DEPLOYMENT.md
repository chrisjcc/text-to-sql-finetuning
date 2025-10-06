# Deployment Guide

This guide covers different deployment strategies for your fine-tuned text-to-SQL model.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Hugging Face Inference Endpoints](#hugging-face-inference-endpoints)
3. [Docker Deployment](#docker-deployment)
4. [API Server Deployment](#api-server-deployment)
5. [Production Considerations](#production-considerations)

## Local Deployment

### Using Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load model
model_path = "./code-llama-3-1-8b-text-to-sql"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate SQL
schema = """CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    city VARCHAR(50)
);"""

messages = [
    {"role": "system", "content": f"You are a text to SQL query translator. SCHEMA:\n{schema}"},
    {"role": "user", "content": "Show me all customers from New York"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
result = pipe(prompt, max_new_tokens=256, do_sample=False)
sql = result[0]['generated_text'][len(prompt):].strip()

print(f"Generated SQL:\n{sql}")
```

## Hugging Face Inference Endpoints

### Step 1: Push Model to Hub

Update `scripts/train.py` to enable pushing to hub:

```python
training_args = trainer.create_training_arguments(
    # ... other args ...
    push_to_hub=True,
)
```

Or push manually after training:

```bash
huggingface-cli login
huggingface-cli upload your-username/text-to-sql-llama ./code-llama-3-1-8b-text-to-sql
```

### Step 2: Create Inference Endpoint

1. Go to https://ui.endpoints.huggingface.co/
2. Click "Create new endpoint"
3. Select your model
4. Choose instance type (e.g., GPU.small for testing)
5. Deploy

### Step 3: Use the Endpoint

```python
import requests

API_URL = "https://your-endpoint.endpoints.huggingface.cloud"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def generate_sql(question, schema):
    payload = {
        "inputs": f"Schema: {schema}\nQuestion: {question}\nSQL:",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.95
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"]
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy model and code
COPY ./code-llama-3-1-8b-text-to-sql ./model
COPY ./src ./src
COPY ./config ./config

# Expose port
EXPOSE 8000

# Run server
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
docker build -t text-to-sql-api .
docker run -p 8000:8000 --gpus all text-to-sql-api
```

## API Server Deployment

### FastAPI Server

Create `server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="Text-to-SQL API")

# Load model at startup
model = None
pipe = None

@app.on_event("startup")
async def load_model():
    global model, pipe
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = "./code-llama-3-1-8b-text-to-sql"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

class SQLRequest(BaseModel):
    question: str
    schema: str
    max_tokens: int = 256

class SQLResponse(BaseModel):
    sql: str
    
@app.post("/generate", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    try:
        messages = [
            {"role": "system", "content": f"You are a text to SQL query translator. SCHEMA:\n{request.schema}"},
            {"role": "user", "content": request.question}
        ]
        
        prompt = pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        result = pipe(
            prompt, 
            max_new_tokens=request.max_tokens, 
            do_sample=False,
            temperature=0.1
        )
        
        sql = result[0]['generated_text'][len(prompt):].strip()
        return SQLResponse(sql=sql)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipe is not None}
```

### Run Server

```bash
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Test API

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show all customers from NYC",
    "schema": "CREATE TABLE customers (id INT, name VARCHAR(100), city VARCHAR(50));"
  }'
```

## Production Considerations

### 1. Performance Optimization

**Model Quantization**
```python
# Use 8-bit quantization for inference
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Batching**
```python
# Process multiple requests together
def batch_generate(questions, schemas, batch_size=4):
    results = []
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_s = schemas[i:i+batch_size]
        # Process batch...
        results.extend(batch_results)
    return results
```

### 2. Monitoring

**Add Logging**
```python
import logging
from prometheus_client import Counter, Histogram

# Metrics
generation_counter = Counter('sql_generations_total', 'Total SQL generations')
generation_duration = Histogram('sql_generation_duration_seconds', 'SQL generation duration')

@generation_duration.time()
def generate_sql_with_metrics(question, schema):
    generation_counter.inc()
    return generate_sql(question, schema)
```

### 3. Error Handling

```python
class SQLGenerationError(Exception):
    pass

def safe_generate_sql(question, schema, max_retries=3):
    for attempt in range(max_retries):
        try:
            sql = generate_sql(question, schema)
            # Validate SQL syntax
            validate_sql(sql)
            return sql
        except Exception as e:
            if attempt == max_retries - 1:
                raise SQLGenerationError(f"Failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 4. Security

**SQL Injection Prevention**
```python
import sqlparse

def validate_and_sanitize_sql(sql):
    # Parse SQL
    parsed = sqlparse.parse(sql)
    
    # Check for dangerous operations
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'GRANT']
    for keyword in dangerous_keywords:
        if keyword in sql.upper():
            raise ValueError(f"Dangerous operation detected: {keyword}")
    
    return sql
```

**Rate Limiting**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_sql_rate_limited(request: SQLRequest):
    # ... generation logic
```

### 5. Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_generate_sql(question_hash, schema_hash):
    return generate_sql(question, schema)

def generate_with_cache(question, schema):
    q_hash = hashlib.md5(question.encode()).hexdigest()
    s_hash = hashlib.md5(schema.encode()).hexdigest()
    return cached_generate_sql(q_hash, s_hash)
```

### 6. Load Balancing

Use NGINX or similar for load balancing multiple instances:

```nginx
upstream text_to_sql {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://text_to_sql;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 7. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-to-sql-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: text-to-sql
  template:
    metadata:
      labels:
        app: text-to-sql
    spec:
      containers:
      - name: api
        image: your-registry/text-to-sql-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: text-to-sql-service
spec:
  selector:
    app: text-to-sql
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Cost Optimization

1. **Use smaller models for simpler queries**
2. **Implement query routing based on complexity**
3. **Use CPU for simple queries, GPU for complex ones**
4. **Cache frequent queries**
5. **Implement request batching**
6. **Use spot instances for non-critical workloads**

## Monitoring Checklist

- [ ] Request latency (p50, p95, p99)
- [ ] GPU utilization
- [ ] Memory usage
- [ ] Error rate
- [ ] SQL validation success rate
- [ ] Cache hit rate
- [ ] Cost per query
- [ ] User satisfaction metrics

## Support

For deployment issues, check:
- Model logs
- GPU memory
- API endpoint health
- Network connectivity
- Authentication tokens

For additional help, refer to:
- [Hugging Face Docs](https://huggingface.co/docs)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
