import os
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from src.backend.models.encoder import BertPath, BERTWrapper


class BiEncoderRetriever:
    def __init__(
        self,
        bert_type: BertPath = BertPath.modern_bert,
        root_folder: Optional[str] = None,
        save_embed_on_disk: bool = False,
    ) -> None:
        self.bert_type = bert_type.value
        self.encoder = BERTWrapper(bert_type)
        self.embedding_folder = tempfile.gettempdir()
        self.save_embed_on_disk = save_embed_on_disk
        if self.save_embed_on_disk:
            self.root_folder = root_folder if root_folder is not None else tempfile.gettempdir()
            os.makedirs(
                os.path.join(self.root_folder, self.bert_type.replace("/", "_") + "/"),
                exist_ok=True,
            )

        self.embedding: Optional[npt.NDArray[np.float32]] = None
        self.doc_names: List[str] = []

    def embed_corpus(self, docs: Dict[str, List[str]]) -> None:
        docs_id = docs["id"]
        docs_content = docs["content"]

        if self.all_files_exist(docs_id):
            self.load_embeddings(docs_id)
            return None

        docs_embed = self.encoder.encode_documents(docs_content)
        if self.save_embed_on_disk:
            self._save_embed_doc(docs_id, docs_embed)
        self.doc_names.extend(docs_id)
        if self.embedding is None:
            self.embedding = docs_embed
        else:
            self.embedding = np.concatenate((self.embedding, docs_embed), axis=0)

    def load_embeddings(self, docs_id: List[str]) -> None:
        for doc_id in docs_id:
            file_path = self._get_file_name(doc_id)
            assert os.path.exists(file_path)
            embed = np.expand_dims(np.load(file_path), axis=0)
            if self.embedding is None:
                self.embedding = embed
            else:
                self.embedding = np.concatenate((self.embedding, embed), axis=0)
            self.doc_names.append(doc_id)

    def _get_file_name(self, doc_id: str) -> str:
        file_path = os.path.join(
            self.root_folder, self.bert_type.replace("/", "_") + "/", doc_id + "_embed.npy"
        )
        return file_path

    def all_files_exist(self, docs_id: List[str]) -> bool:
        files_path = [self._get_file_name(doc_id) for doc_id in docs_id]
        return all([os.path.exists(file_path) for file_path in files_path])

    def _save_embed_doc(self, docs_name: List[str], docs_embed: npt.NDArray[np.float32]) -> None:
        # raise NotImplementedError
        for name, doc_embed in zip(docs_name, docs_embed):
            np.save(self._get_file_name(name), doc_embed)

    def _chunk_data(self, data: Dict[str, str], num_chunks: int) -> List[Dict[str, str]]:
        """Split data into approximately equal chunks."""
        keys_chunks = [list(data.keys())[i::num_chunks] for i in range(num_chunks)]
        values_chunks = [list(data.values())[i::num_chunks] for i in range(num_chunks)]
        return [
            {k: v for k, v in zip(keys, values)} for keys, values in zip(keys_chunks, values_chunks)
        ]

    def retrieve(self, query: str) -> List[str]:
        query_embedding = self.encoder.encode_querys([query])
        similarity_scores = self._score(query_embedding)
        return cast(List[str], (np.array(self.doc_names)[np.argsort(-similarity_scores)]).tolist())

    def _score(self, query_embedding: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return query_embedding @ self.embedding.T  # Example dot product


if __name__ == "__main__":
    queries = ["What is TSNE?", "Who is Laurens van der Maaten?", "What color is the horse?"]
    documents = [
        ("TSNE", "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"),
        ("horse", "The horse is white."),
    ]

    bi_encoder_retriver = BiEncoderRetriever()
    bi_encoder_retriver.embed_corpus(documents)

    for query in queries:
        print(bi_encoder_retriver.retrieve(query)[0])
