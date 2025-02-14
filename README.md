# **ModernBERT RAG**

This repository packages the **ModernBERT** bi-encoder as an API for **Retrieval-Augmented Generation (RAG)**.
The provided use case is based on the **Mistral AI documentation**, downloaded from:
🔗 [Mistral Docs](https://docs.mistral.ai)

## 🚀 **How It Works**

### **Architecture**
- **Backend**: A FastAPI-based API that handles encoding, retrieval, and LLM inference.
- **Frontend**: A Chainlit-based UI for user interaction.

## 🛠️ **Installation**

### **1. Create and Activate Conda Environment**
```sh
conda create -n modern_bert python=3.10
conda activate modern_bert
```

### **2. Install Dependencies**
```sh
poetry install
```

## 🔥 **Running the Servers**

### **Frontend (Chainlit UI)**
Launch front end server:
```sh
make frontend
```

### **Backend (FastAPI)**
Launch backend server:
```sh
make backend
```

---

## ✅ **Main Features**
- [x] Working Encoder
- [x] Working Retriever
- [x] Integrated LLM
- [x] Fully Functional RAG Pipeline
- [x] Exposed API
- [x] Interactive Frontend
- [x] Add Encoder Benchmark

---

## 🔄 **Planned Improvements**

### **🔹 Encoder Enhancements**
- [x] Improve dataset structure support
- [x] Add a second encoder for comparison

### **🔹 RAG Improvements**
- [ ] Add LLM pre-processing for better query generation

### **🔹 Retriver Improvements**
- [x] Add Faiss for vector storage

---

## 📡 **API Usage**

### **Base URL**
```
http://localhost:8000/
```

### **Endpoints**

#### 🔹 **1. Query the RAG System**
**Endpoint:**
```http
POST /query
```
**Request Body:**
```json
{
  "question": "What is Mistral?", "use_llm": true, "retrieve_n_docs": 1
}
```
**Response:**
```json
{
  "answer": "Mistral, based on the provided context, is a company that develops and releases various models, including text and image understanding models, open-source models, and a math model. It also offers APIs for text generation, vision analysis, code generation, embeddings, function calling, fine-tuning, JSON mode, and guardrailing."
}
```

---

## 🎯 **Example Usage**
You can interact with the API using **cURL**, **Python (requests/httpx)**, or **Postman**.

### **Using `httpx` in Python**
```python
import httpx

response = httpx.post("http://localhost:8000/query", json={"question": "What is ModernBERT?"})
print(response.json())
```
## Download Corpus
You can try download your own corpus with the command

```sh
wget --recursive --no-parent --convert-links --page-requisites --domains docs.mistral.ai https://docs.mistral.ai
```

and the filter it and clean it with
```sh
python scripts.filter_corpus.py <path_to_the_corpus>
```


## 🚀 **Benchmark**

You can find the benchmark code [here](src/backend/models/benchmark.py).

### **Performance**

#### **L4**
- **Max Chunk 512 (batch size 600)**
  - ModernBERT Base: **38K tokens/sec**
  - GTE Base: **46K tokens/sec**
- **Max Chunk 8192 (batch size 20)**
  - ModernBERT Base: **8K tokens/sec**
  - GTE Base: **19K tokens/sec**

#### **A Series**
- **Max Chunk 512 (batch size 600)**
  - ModernBERT Base: **1091 tokens/sec**
  - GTE Base: **715K tokens/sec**

- **Max Chunk 8192 (batch size 100)**
  - ModernBERT Base: **2084K tokens/sec**
  - GTE Base: **221K tokens/sec**


## 🛠️ Development Commands

### Code Formatting & Style Checks
- **Run Style Checks:**

Runs `black` `mypy` and `flake8` to enforce code style.
```sh
make cheks
```

- **Run Style Checks:**

This repository use pre commit, you can install it with:

```sh
make pre-commit-install
```
