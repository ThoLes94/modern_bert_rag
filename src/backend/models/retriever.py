import os
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from src.backend.models.encoder import BertHFPath, BERTWrapperHF

if True:
    # Need to import faiss after to avoid segmentation fault error ....
    import faiss


class BiEncoderRetriever:
    def __init__(
        self,
        bert_type: BertHFPath = BertHFPath.modern_bert_large_embed,
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
            docs_embed_normalized = docs_embed / np.linalg.norm(docs_embed, axis=0)
            self.docs_id.extend(docs_id)
            self.faiss_index.add(docs_embed_normalized)

            if self.embeddings is None:
                self.embeddings = docs_embed
            else:
                self.embeddings = np.concatenate((self.embeddings, docs_embed), axis=0)
            if self.save_load_embed_on_disk:
                self._save_embed_doc(docs_id, docs_embed)

    def load_embeddings(self, docs_id: List[str]) -> None:
        for doc_id in docs_id:
            file_path = self._get_file_name(doc_id)
            assert os.path.exists(file_path)
            embed = np.expand_dims(np.load(file_path), axis=0)
            embed_normalized = embed / np.linalg.norm(embed, axis=0)
            self.faiss_index.add(embed_normalized)
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
        query_embedding = self.encoder.encode_queries([query])
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)

        _, indices = self.faiss_index.search(query_embedding_normalized, retun_n_doc)
        # Print the most similar documents
        return [self.docs_id[i] for i in indices[0]]

    def _score(self, queries_embedding: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert (
            self.embeddings is not None
        ), "Corpus embedding has not been generated, call `embed_corpus`"
        return queries_embedding @ self.embeddings.T  # Example dot product


if __name__ == "__main__":
    documents = {
        "id": ["Emmanuel Macron", "TSNE", "horse", "first", "second"],
        "content": [
            "Emmanuel Macron was born on December 21, 1977, in Amiens, France. He is a French politician who served as the President of France from 2017 to 2022. Before his presidency, he worked as an investment banker and was an inspector of finances for the French government.",
            "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
            "The horse is white.",
            "The first-line therapy for patients with metastatic pancreatic cancer is FOLFIRINOX",
            "The second-line therapy for patients with metastatic pancreatic cancer is Gemzar-Abraxane",
        ],
    }

    queries = [
        "When is born Emannuel Macron?",
        "What color is the horse?",
        "What is TSNE?",
        "What is the second line of the metastatic pancreatic cancer?",
        "What is the first line of the metastatic pancreatic cancer?",
    ]

    bi_encoder_retriver = BiEncoderRetriever(BertHFPath.gte_base)
    bi_encoder_retriver.embed_corpus(documents)

    # print(bi_encoder_retriver.embeddings.shape)
    for query in queries:
        print(bi_encoder_retriver.retrieve(query)[0])
