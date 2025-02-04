import os
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from src.backend.models.encoder import BertHFPath, BERTWrapperHF

if True:
    # Need to import faiss after to avoid segment fault error ....
    import faiss


class BiEncoderRetriever:
    def __init__(
        self,
        bert_type: BertHFPath = BertHFPath.modern_bert_large,
        root_folder: str = "data/embedding",
        save_load_embed_on_disk: bool = False,
    ) -> None:

        self.bert_type = bert_type.value
        self.encoder = BERTWrapperHF(bert_type)
        self.save_load_embed_on_disk = save_load_embed_on_disk
        dim = self.encoder.model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(dim)
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

        else:
            docs_embed = self.encoder.encode_documents(docs_content)
            self.docs_id.extend(docs_id)
            self.faiss_index.add(docs_embed)

            if self.embeddings is None:
                self.embeddings = docs_embed
            else:
                self.embeddings = np.concatenate((self.embeddings, docs_embed), axis=0)

    def load_embeddings(self, docs_id: List[str]) -> None:
        for doc_id in docs_id:
            file_path = self._get_file_name(doc_id)
            assert os.path.exists(file_path)
            embed = np.expand_dims(np.load(file_path), axis=0)
            self.faiss_index.add(embed)
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

    def retrieve(self, query: str, retun_n_doc: int = 2) -> List[str]:
        query_embedding = self.encoder.encode_querys([query])

        distances, indices = self.faiss_index.search(query_embedding, retun_n_doc)

        # Print the most similar documents
        return [self.docs_id[i] for i in indices[0]]

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
    bi_encoder_retriver = BiEncoderRetriever(BertHFPath.modern_bert_large)
    bi_encoder_retriver.embed_corpus(documents)

    # print(bi_encoder_retriver.embeddings.shape)
    for query in queries:
        print(bi_encoder_retriver.retrieve(query)[0])
