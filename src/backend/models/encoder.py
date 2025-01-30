from enum import Enum
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer


class BertPath(str, Enum):
    modern_bert = "lightonai/modernbert-embed-large"


class BERTWrapper:
    def __init__(self, model_path: BertPath = BertPath.modern_bert) -> None:
        self.model = SentenceTransformer(model_path.value)

    def encode_prompt(self, texts: List[str]) -> npt.NDArray[np.float32]:
        query_embeddings = self.model.encode([f"search_query: {txt}" for txt in texts])
        return query_embeddings

    def encode_document(self, docs: List[str]) -> npt.NDArray[np.float32]:
        doc_embeddings = self.model.encode([f"search_document: {txt}" for txt in docs])
        return doc_embeddings

    def get_similarity(
        self, query_embeddings: npt.NDArray[np.float32], doc_embeddings: npt.NDArray[np.float32]
    ) -> torch.Tensor:
        similarities = self.model.similarity(query_embeddings, doc_embeddings)
        return similarities


if __name__ == "__main__":
    bert_wrapper = BERTWrapper(BertPath.modern_bert)

    query_embeddings = bert_wrapper.encode_prompt(
        [
            "What is TSNE?",
            "Who is Laurens van der Maaten?",
        ]
    )
    doc_embeddings = bert_wrapper.encode_document(
        [
            "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
        ]
    )
    print(query_embeddings.shape, doc_embeddings.shape)
    # (2, 1024) (1, 1024)

    similarities = bert_wrapper.get_similarity(query_embeddings, doc_embeddings)
    print(similarities)
    # tensor([[0.6518],
    #         [0.4237]])
