# from chainlit.utils import mount_chainlit
from typing import Dict, Iterable, List

from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader

from src.backend.corpus.corpus import DatasetWrapper
from src.backend.models.encoder import BertPath
from src.backend.models.llm import LLMType
from src.backend.models.rag import RAGWrapper

app = FastAPI()

# Initialize RAG system
rag_wrapper = RAGWrapper(BertPath.modern_bert, llm_type=LLMType.mistral_7b)

# Prepare corpus at startup
corpus = DatasetWrapper("src/backend/corpus/docs.mistral.ai")

dataloader: Iterable[Dict[str, List[str]]] = DataLoader(corpus, batch_size=1)

rag_wrapper.prepare_corpus(dataloader)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_rag(request: QueryRequest) -> Dict[str, str]:
    """API endpoint to get answers from RAG."""
    response = rag_wrapper.answer_question(request.question)
    return {"answer": response}


# mount_chainlit(app=app, target="src/frontend/my_cl_app.py", path="/chainlit")
