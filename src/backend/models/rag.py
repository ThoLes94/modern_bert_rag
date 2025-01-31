from typing import Iterable, Tuple

from src.backend.models.encoder import BertPath
from src.backend.models.llm import LLMType, LLMWrapper
from src.backend.models.retriever import BiEncoderRetriever


class RAGWrapper:
    def __init__(self, encoder_type: BertPath, llm_type: LLMType) -> None:
        self.retriver = BiEncoderRetriever(encoder_type)
        self.llm = LLMWrapper(llm_type)
        self._initialize_llm()

    def prepare_corpus(self, corpus: Iterable[Tuple[str, str]]) -> None:
        self.corpus = {name: doc for name, doc in corpus}
        self.retriver.embed_corpus(corpus)

    def _initialize_llm(self) -> None:
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
        best_document = self.corpus[best_document_name]

        prompt = self.format_prompt(best_document, question)

        llm_output = self.llm.generate_answer(prompt=prompt)
        return llm_output.last_answer


if __name__ == "__main__":
    corpus = [
        (
            "Emmanuel Macron",
            "Emmanuel Macron was born on December 21, 1977, in Amiens, France. He is a French politician who served as the President of France from 2017 to 2022. Before his presidency, he worked as an investment banker and was an inspector of finances for the French government.",
        ),
        ("TSNE", "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"),
        ("horse", "The horse is white."),
        (
            "first line treatement",
            "The first-line therapy for patients with metastatic pancreatic cancer is FOLFIRINOX",
        ),
        (
            "second line treatement",
            "The second-line therapy for patients with metastatic pancreatic cancer is Gemzar-Abraxane",
        ),
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
