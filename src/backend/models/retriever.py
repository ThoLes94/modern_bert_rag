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
        self.encoder = BERTWrapper(bert_type)
        self.embedding_folder = tempfile.gettempdir()
        self.save_embed_on_disk = save_embed_on_disk
        if self.save_embed_on_disk:
            self.root_folder = root_folder if root_folder is not None else tempfile.gettempdir()

        self.embedding: List[npt.ArrayLike] = []
        self.doc_names: List[str] = []

    def embed_corpus(self, corpus: Iterable[Tuple[str, str]]) -> None:
        for doc_name, doc in corpus:
            doc_embed = self.encoder.encode_documents([doc])
            if self.save_embed_on_disk:
                self._save_embed_doc(doc_name, doc_embed)
            else:
                self.doc_names.append(doc_name)
                self.embedding.append(doc_embed)

    def _save_embed_doc(self, doc_name: str, doc_embed: npt.ArrayLike) -> None:
        np.save(os.path.join(self.root_folder, doc_name + "_embed.npy"), doc_embed)

    def _chunk_data(self, data: Dict[str, str], num_chunks) -> List[Dict[str, str]]:
        """Split data into approximately equal chunks."""
        keys_chunks = [list(data.keys())[i::num_chunks] for i in range(num_chunks)]
        values_chunks = [list(data.values())[i::num_chunks] for i in range(num_chunks)]
        return [
            {k: v for k, v in zip(keys, values)} for keys, values in zip(keys_chunks, values_chunks)
        ]

    def retrieve(self, query) -> List[str]:
        query_embedding = self.encoder.encode_querys([query])
        similarity_scores = self._score(query_embedding)
        return cast(List[str], (np.array(self.doc_names)[np.argsort(-similarity_scores)]).tolist())

    def _score(self, query_embedding):
        return query_embedding @ np.concat(self.embedding).T  # Example dot product


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
