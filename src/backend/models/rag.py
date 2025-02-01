from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backend.corpus.corpus import DatasetWrapper
from src.backend.models.encoder import BertPath
from src.backend.models.llm import LLMType, LLMWrapper
from src.backend.models.retriever import BiEncoderRetriever


class RAGWrapper:
    def __init__(self, encoder_type: BertPath, llm_type: LLMType, use_llm: bool = False) -> None:
        self.llm_type = llm_type
        self.retriver = BiEncoderRetriever(
            encoder_type, save_embed_on_disk=True, root_folder="data/"
        )
        self.use_llm = use_llm

    def prepare_corpus(self, corpus: Iterable[Dict[str, List[str]]]) -> None:
        self.corpus: Dict[str, str] = {}
        for docs in tqdm(corpus):
            self.corpus.update(
                {doc_id: doc_content for doc_id, doc_content in zip(docs["id"], docs["content"])}
            )
            self.retriver.embed_corpus(docs)
        if self.use_llm:
            self._initialize_llm()

    def _initialize_llm(self) -> None:
        self.llm = LLMWrapper(self.llm_type)
        initial_message = """You are a helpful chatbot. You will be provided each time a document and a question.
            Use only the following pieces of context to answer the question. Don't make up any new information!"""
        _ = self.llm.generate_answer(initial_message, None)

    def format_prompt(self, retrieved_chunk: str, question: str) -> str:
        prompt = f"""
        Context information is below.
        ---------------------
        {retrieved_chunk}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Answer:
        """
        return prompt

    def answer_question(self, question: str) -> str:
        best_document_name = self.retriver.retrieve(question)[0][0]
        # second_best_document_name = self.retriver.retrieve(question)[0][1]
        best_document = self.corpus[best_document_name]
        # second_best_document = self.corpus[second_best_document_name]
        if not self.use_llm:
            return best_document

        prompt = self.format_prompt(best_document, question)

        llm_output = self.llm.generate_answer(prompt=prompt)
        return llm_output.last_answer


if __name__ == "__main__":
    corpus = [
        {
            "id": ["Emmanuel Macron", "TSNE", "horse", "first", "second"],
            "content": [
                "Emmanuel Macron was born on December 21, 1977, in Amiens, France. He is a French politician who served as the President of France from 2017 to 2022. Before his presidency, he worked as an investment banker and was an inspector of finances for the French government.",
                "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
                "The horse is white.",
                "The first-line therapy for patients with metastatic pancreatic cancer is FOLFIRINOX",
                "The second-line therapy for patients with metastatic pancreatic cancer is Gemzar-Abraxane",
            ],
        }
    ]
    rag_wrapper = RAGWrapper(BertPath.modern_bert, llm_type=LLMType.mistral_7b)
    rag_wrapper.prepare_corpus(corpus)

    queries = [
        "When is born Emannuel Macron?",
        "What color is the horse?",
        "What is TSNE?",
        "What is the second line of the metastatic pancreatic cancer?",
        "What is the first line of the metastatic pancreatic cancer?",
    ]

    for query in queries:
        print(rag_wrapper.answer_question(query))
