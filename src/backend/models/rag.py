from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from src.backend.corpus.corpus import DatasetWrapper
from src.backend.models.encoder import BertHFPath
from src.backend.models.llm import LLMHFPath, LLMWrapper
from src.backend.models.retriever import BiEncoderRetriever


class RAGWrapper:
    def __init__(
        self,
        encoder_path: BertHFPath,
        llm_type: LLMHFPath,
        corpus_path: str,
        chunk_size: int = 2048,
    ) -> None:
        self.llm_type = llm_type
        self.retriever = BiEncoderRetriever(
            encoder_path, save_load_embed_on_disk=True, root_folder="data/embedding/"
        )
        self.path_to_file = corpus_path
        self.chunk_size = chunk_size
        self.encoder = encoder_path
        self.dataset = DatasetWrapper(self.path_to_file)
        self._prepare_corpus()
        self._initialize_llm()

    def _prepare_corpus(self) -> None:
        self.corpus: Dict[str, str] = {}
        dataloader = self.dataset.get_dataloader(
            batch_size=10, tokenizer_path=self.encoder, chunk_size=self.chunk_size
        )
        for docs in tqdm(dataloader):
            self.corpus.update(
                {doc_id: doc_content for doc_id, doc_content in zip(docs["id"], docs["content"])}
            )
            self.retriever.embed_corpus(docs)

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

    def answer_question(
        self,
        question: str,
        encoder_path: BertHFPath,
        chunk_size: int,
        use_llm: bool = False,
        return_n_doc: int = 1,
    ) -> str:
        if self.retriever.bert_type != encoder_path.value or chunk_size != self.chunk_size:
            self._update_encoder_chunk_size(encoder_path, chunk_size)
        documents = "\n".join(
            [
                self.corpus[document_name]
                for document_name in self.retriever.retrieve(question, retun_n_doc=return_n_doc)
            ]
        )
        if not use_llm:
            return documents

        prompt = self._format_prompt(documents, question)

        llm_output = self.llm.generate_answer(prompt=prompt)
        return llm_output.last_answer

    def _update_encoder_chunk_size(self, encoder_path: BertHFPath, chumk_size: int) -> None:
        self.retriever = BiEncoderRetriever(
            encoder_path, save_load_embed_on_disk=True, root_folder="data/embedding/"
        )
        self.encoder = encoder_path
        self.chunk_size = chumk_size
        self._prepare_corpus()


if __name__ == "__main__":
    # Initialize RAG system
    rag_wrapper = RAGWrapper(
        BertHFPath.modern_bert_base,
        llm_type=LLMHFPath.mistral_7b,
        corpus_path="data/corpus/docs.mistral.ai",
        chunk_size=2048,
    )

    rag_wrapper._update_encoder_chunk_size(BertHFPath.modern_bert_base_embed, 2048)

    queries = [
        "When is born Emannuel Macron?",
        "What color is the horse?",
        "What is TSNE?",
        "What is the second line of the metastatic pancreatic cancer?",
        "What is the first line of the metastatic pancreatic cancer?",
    ]

    for query in queries:
        print(
            rag_wrapper.answer_question(
                query,
                BertHFPath.modern_bert_base_embed,
                2048,
                return_n_doc=2,
            )
        )
