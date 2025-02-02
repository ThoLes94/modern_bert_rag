# from chainlit.utils import mount_chainlit
from typing import Dict, Iterable, List

from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader

from src.backend.corpus.corpus import DatasetWrapper
from src.backend.models.encoder import BertHFPath
from src.backend.models.llm import LLMHFPath
from src.backend.models.rag import RAGWrapper

app = FastAPI()

# Initialize RAG system

# Prepare corpus at startup

corpus = DatasetWrapper("data/corpus/docs.mistral.ai")
dataloader: Iterable[Dict[str, List[str]]] = DataLoader(corpus, batch_size=1)
rag_wrapper = RAGWrapper(
    BertHFPath.modern_bert_large, llm_type=LLMHFPath.mistral_7b, corpus=dataloader
)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_rag(request: QueryRequest) -> Dict[str, str]:
    """API endpoint to get answers from RAG."""
    response = rag_wrapper.answer_question(request.question)
    return {"answer": response}


@app.post("/deactivate_llm")
async def deactivate_llm() -> Dict[str, bool]:
    """API endpoint to get answers from RAG."""
    rag_wrapper.use_llm = False
    return {"llm_state": False}


@app.post("/activate_llm")
async def acivate_llm() -> Dict[str, bool]:
    """API endpoint to get answers from RAG."""
    rag_wrapper.use_llm = True
    return {"llm_state": True}
