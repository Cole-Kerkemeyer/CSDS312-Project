# Tumor Classification and Segmentation

## LLM Integration

This project now includes a lightweight OpenAI-compatible client in `llm_client.py`.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Set environment variables

```bash
# Required
set OPENAI_API_KEY=your_api_key

# Optional: default model
set OPENAI_MODEL=Qwen2.5-7B-Instruct

# Optional: OpenAI-compatible provider endpoint (example: Alibaba DashScope)
set OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3) Quick test

```bash
python llm_client.py
```

### 4) Use in your scripts

```python
from llm_client import LLMClient

llm = LLMClient(model="Qwen2.5-7B-Instruct")
result = llm.chat("Explain overfitting in 3 bullet points.")
print(result)
```
