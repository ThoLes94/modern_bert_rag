import os
import tempfile
from typing import Dict, List, Optional, cast

import numpy as np
import numpy.typing as npt

from src.backend.models.encoder import BertHFPath, BERTWrapperHF


class BiEncoderRetriever:
    def __init__(
        self,
        bert_type: BertHFPath = BertHFPath.modern_bert_large,
        root_folder: str = "data/embedding",
        save_load_embed_on_disk: bool = False,
    ) -> None:
        self.bert_type = bert_type.value
        self.encoder = BERTWrapperHF(bert_type)
        self.embedding_folder = tempfile.gettempdir()
        self.save_load_embed_on_disk = save_load_embed_on_disk
        self.root_folder = root_folder
        if self.save_load_embed_on_disk:
            os.makedirs(
                os.path.join(self.root_folder, self.bert_type.replace("/", "_") + "/"),
                exist_ok=True,
            )

        self.embeddings: Optional[npt.NDArray[np.float32]] = None
        self.docs_id: List[str] = []

    def embed_corpus(self, docs: Dict[str, List[str]]) -> None:
        docs_id = docs["id"]
        docs_content = docs["content"]

        if self.save_load_embed_on_disk and self._all_files_exist(docs_id):
            self.load_embeddings(docs_id)
            return None

        docs_embed = self.encoder.encode_documents(docs_content)
        if self.save_load_embed_on_disk:
            self._save_embed_doc(docs_id, docs_embed)
        self.docs_id.extend(docs_id)
        if self.embeddings is None:
            self.embeddings = docs_embed
        else:
            self.embeddings = np.concatenate((self.embeddings, docs_embed), axis=0)

    def load_embeddings(self, docs_id: List[str]) -> None:
        for doc_id in docs_id:
            file_path = self._get_file_name(doc_id)
            assert os.path.exists(file_path)
            embed = np.expand_dims(np.load(file_path), axis=0)
            if self.embeddings is None:
                self.embeddings = embed
            else:
                self.embeddings = np.concatenate((self.embeddings, embed), axis=0)
            self.docs_id.append(doc_id)

    def _get_file_name(self, doc_id: str) -> str:
        file_path = os.path.join(
            self.root_folder, self.bert_type.replace("/", "_") + "/", doc_id + "_embed.npy"
        )
        return file_path

    def _all_files_exist(self, docs_id: List[str]) -> bool:
        files_path = [self._get_file_name(doc_id) for doc_id in docs_id]
        return all([os.path.exists(file_path) for file_path in files_path])

    def _save_embed_doc(self, docs_name: List[str], docs_embed: npt.NDArray[np.float32]) -> None:
        # raise NotImplementedError
        for name, doc_embed in zip(docs_name, docs_embed):
            np.save(self._get_file_name(name), doc_embed)

    def retrieve(self, query: str) -> List[str]:
        query_embedding = self.encoder.encode_querys([query])
        similarity_scores = self._score(query_embedding)
        return cast(List[str], (np.array(self.docs_id)[np.argsort(-similarity_scores)]).tolist())

    def _score(self, queries_embedding: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert (
            self.embeddings is not None
        ), "Corpus embedding has not been generated, call `embed_corpus`"
        return queries_embedding @ self.embeddings.T  # Example dot product


if __name__ == "__main__":
    queries = ["What is TSNE?", "Who is Laurens van der Maaten?", "What color is the horse?"]
    documents = {
        "id": ["TSNE", "horse"],
        "content": [
            "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
            "The horse is white.",
        ],
    }

    bi_encoder_retriver = BiEncoderRetriever()
    bi_encoder_retriver.embed_corpus(documents)

    for query in queries:
        print(bi_encoder_retriver.retrieve(query)[0])
