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
```sh
make frontend
```

### **Backend (FastAPI)**
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
- [ ] Add Encoder Evaluation

---

## 🔄 **Planned Improvements**

### **🔹 Encoder Enhancements**
- [ ] Implement multi-processing for faster corpus embedding
- [x] Improve dataset structure support
- [ ] Add a second encoder for comparison

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
