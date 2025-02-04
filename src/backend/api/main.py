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

# Prepare corpus at startup
corpus = DatasetWrapper("data/corpus/docs.mistral.ai", chunk_size=2048)
dataloader: Iterable[Dict[str, List[str]]] = DataLoader(corpus, batch_size=1)

# Initialize RAG system
rag_wrapper = RAGWrapper(
    BertHFPath.modern_bert_base, llm_type=LLMHFPath.mistral_7b, corpus=dataloader
)


class QueryRequest(BaseModel):
    question: str
    use_llm: bool = False
    return_n_doc: int = 4


@app.post("/query")
async def query_rag(request: QueryRequest) -> Dict[str, str]:
    """API endpoint to get answers from RAG."""
    response = rag_wrapper.answer_question(
        request.question, use_llm=request.use_llm, return_n_doc=request.return_n_doc
    )
    return {"answer": response}
