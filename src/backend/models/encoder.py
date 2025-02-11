from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class BertHFPath(str, Enum):
    modern_bert_large = "lightonai/modernbert-embed-large"
    modern_bert_base = "nomic-ai/modernbert-embed-base"
    gte_large = "Alibaba-NLP/gte-large-en-v1.5"


class BERTWrapperHF:
    def __init__(
        self,
        model_path: BertHFPath = BertHFPath.modern_bert_large,
        device: str = "mps",
    ) -> None:
        self.model = SentenceTransformer(model_path.value, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.value)
        self.model.to(device)

    def encode_queries(self, texts: List[str]) -> npt.NDArray[np.float32]:
        with torch.no_grad():
            query_embeddings: npt.NDArray[np.float32] = self.model.encode(
                [f"search_query: {txt}" for txt in texts]
            )
        return query_embeddings

    def encode_documents(self, docs: List[str]) -> npt.NDArray[np.float32]:
        # TODO: verify numbers of tokens?
        with torch.no_grad():
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                doc_embeddings: npt.NDArray[np.float32] = self.model.encode(
                    [f"search_document: {txt}" for txt in docs]
                )
        return doc_embeddings

    def get_similarity(
        self, query_embeddings: npt.NDArray[np.float32], doc_embeddings: npt.NDArray[np.float32]
    ) -> torch.Tensor:
        similarities: torch.Tensor = self.model.similarity(query_embeddings, doc_embeddings)
        return similarities


if __name__ == "__main__":
    bert_wrapper = BERTWrapperHF(BertHFPath.gte_large)

    query_embeddings = bert_wrapper.encode_queries(
        ["What is TSNE?", "Who is Laurens van der Maaten?", "What color is the horse"]
    )
    doc_embeddings = bert_wrapper.encode_documents(
        [
            "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
            "The horse is white",
        ]
    )
    print(query_embeddings.shape, doc_embeddings.shape)
    # (2, 1024) (1, 1024)

    similarities = bert_wrapper.get_similarity(query_embeddings, doc_embeddings)
    print(similarities)
    # tensor([[0.6518],
    #         [0.4237]])
