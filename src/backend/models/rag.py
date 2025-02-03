from typing import Dict, Iterable, List

from tqdm import tqdm

from src.backend.models.encoder import BertHFPath
from src.backend.models.llm import LLMHFPath, LLMWrapper
from src.backend.models.retriever import BiEncoderRetriever


class RAGWrapper:
    def __init__(
        self,
        encoder_type: BertHFPath,
        llm_type: LLMHFPath,
        corpus: Iterable[Dict[str, List[str]]],
    ) -> None:
        self.llm_type = llm_type
        self.retriver = BiEncoderRetriever(
            encoder_type, save_load_embed_on_disk=True, root_folder="data/"
        )
        self._prepare_corpus(corpus)
        self._initialize_llm()

    def _prepare_corpus(self, corpus: Iterable[Dict[str, List[str]]]) -> None:
        self.corpus: Dict[str, str] = {}
        for docs in tqdm(corpus):
            self.corpus.update(
                {doc_id: doc_content for doc_id, doc_content in zip(docs["id"], docs["content"])}
            )
            self.retriver.embed_corpus(docs)

    def _initialize_llm(self) -> None:
        self.llm = LLMWrapper(self.llm_type)
        initial_message = """You are a helpful chatbot. You will be provided each time a document and a question.
            Use only the following pieces of context to answer the question. Don't make up any new information!"""
        _ = self.llm.generate_answer(initial_message, None)

    def _format_prompt(self, retrieved_chunk: str, question: str) -> str:
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

    def answer_question(self, question: str, use_llm: bool = False, return_n_doc: int = 1) -> str:
        documents = "\n".join(
            [
                self.corpus[document_name]
                for document_name in self.retriver.retrieve(question)[0][:return_n_doc]
            ]
        )
        if not use_llm:
            return documents

        prompt = self._format_prompt(documents, question)

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
    rag_wrapper = RAGWrapper(
        BertHFPath.modern_bert_large, llm_type=LLMHFPath.mistral_7b, corpus=corpus
    )
    # rag_wrapper.use_llm = False
    # rag_wrapper.use_llm = True

    queries = [
        "When is born Emannuel Macron?",
        "What color is the horse?",
        "What is TSNE?",
        "What is the second line of the metastatic pancreatic cancer?",
        "What is the first line of the metastatic pancreatic cancer?",
    ]

    for query in queries:
        print(rag_wrapper.answer_question(query))
