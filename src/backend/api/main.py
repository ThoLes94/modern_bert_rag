# from chainlit.utils import mount_chainlit
import json
import logging
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.backend.models.benchmark import benchmark_on_corpus
from src.backend.models.encoder import BertHFPath
from src.backend.models.llm import LLMHFPath
from src.backend.models.rag import RAGWrapper

# Configure logging
logging.basicConfig(filename="query_logs.log", level=logging.INFO)


app = FastAPI()

# Initialize RAG system
rag_wrapper = RAGWrapper(
    BertHFPath.modern_bert_base_embed,
    llm_type=LLMHFPath.mistral_7b,
    corpus_path="data/corpus/docs.mistral.ai",
    chunk_size=2048,
)


class QueryRequest(BaseModel):
    question: str
    use_llm: bool = False
    return_n_doc: int = 4
    encoder: BertHFPath = BertHFPath.modern_bert_base_embed
    chunk_size: int = 2048


@app.post("/query")
async def query_rag(request: QueryRequest) -> Dict[str, str]:
    """API endpoint to get answers from RAG."""
    response = rag_wrapper.answer_question(
        request.question,
        use_llm=request.use_llm,
        return_n_doc=request.return_n_doc,
        encoder_path=request.encoder,
        chunk_size=request.chunk_size,
    )
    return {"answer": response}


class BenchmarkRequest(BaseModel):
    encoder_path: BertHFPath
    batch_size: int = 10
    chunk_size: int = 512
    num_pass: int = 1


@app.post("/benchmark_corpus")
async def run_benchmark(request: BenchmarkRequest) -> Dict[str, str]:
    """API endpoint to get answers from RAG."""

    results = benchmark_on_corpus(
        request.encoder_path,
        batch_size=request.batch_size,
        chunk_size=request.chunk_size,
        num_pass=request.num_pass,
    )
    logging.info(json.dumps(results))

    return {"answer": json.dumps(results)}
