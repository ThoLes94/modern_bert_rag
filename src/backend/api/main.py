# from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from pydantic import BaseModel

from src.backend.models.encoder import BertPath
from src.backend.models.llm import LLMType
from src.backend.models.rag import RAGWrapper

app = FastAPI()

# Initialize RAG system
rag_wrapper = RAGWrapper(BertPath.modern_bert, llm_type=LLMType.mistral_7b)

# Prepare corpus at startup
corpus = [
    ("Emmanuel Macron", "Emmanuel Macron was born on December 21, 1977, in Amiens, France."),
    ("TSNE", "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"),
    ("horse", "The horse is white."),
    (
        "first line treatment",
        "The first-line therapy for metastatic pancreatic cancer is FOLFIRINOX",
    ),
    (
        "second line treatment",
        "The second-line therapy for metastatic pancreatic cancer is Gemzar-Abraxane",
    ),
]
rag_wrapper.prepare_corpus(corpus)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_rag(request: QueryRequest):
    """API endpoint to get answers from RAG."""
    response = rag_wrapper.answer_question(request.question)
    return {"answer": response}


# mount_chainlit(app=app, target="src/frontend/my_cl_app.py", path="/chainlit")
