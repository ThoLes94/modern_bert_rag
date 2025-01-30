import os
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer


def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Assign GPU to process


class BertPath(str, Enum):
    modern_bert = "lightonai/modernbert-embed-large"


class BERTWrapper(torch.nn.Module):
    def __init__(self, model_path: BertPath = BertPath.modern_bert) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_path.value)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.value)

    def encode_prompt(self, texts: List[str]) -> npt.NDArray[np.float32]:
        with torch.no_grad():
            query_embeddings = self.model.encode([f"search_query: {txt}" for txt in texts])
        return query_embeddings

    def encode_document(self, docs: List[str]) -> npt.NDArray[np.float32]:
        # TODO: verify numbers of tokens?
        with torch.no_grad():
            doc_embeddings = self.model.encode([f"search_document: {txt}" for txt in docs])
        return doc_embeddings

    def get_similarity(
        self, query_embeddings: npt.NDArray[np.float32], doc_embeddings: npt.NDArray[np.float32]
    ) -> torch.Tensor:
        similarities = self.model.similarity(query_embeddings, doc_embeddings)
        return similarities

    def forward(self, x: Any):
        return self.model(**x)

    def _chunk_data(self, data: Dict[str, str], num_chunks):
        """Split data into approximately equal chunks."""
        keys_chunks = [list(data.keys())[i::num_chunks] for i in range(num_chunks)]
        values_chunks = [list(data.values())[i::num_chunks] for i in range(num_chunks)]
        return [
            {k: v for k, v in zip(keys, values)} for keys, values in zip(keys_chunks, values_chunks)
        ]

    def encode_corpus(self, corpus: Dict[str, str]) -> None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            doc_chunks = self._chunk_data(corpus, world_size)  # Split documents for each GPU
            self.embeddings = {}

            queue = mp.Queue()
            mp.spawn(
                self.encode_chunk_docs,
                args=(world_size, doc_chunks, queue),
                nprocs=world_size,
                join=True,
            )
            while not queue.empty():
                self.embeddings.update(queue.get())
        else:
            for name, doc in corpus.items():
                self.embeddings[name] = self.encode_document([doc])

    def encode_chunk_docs(self, rank, world_size, doc_chunks, queue):
        setup_ddp(rank, world_size)

        device = torch.device(f"cuda:{rank}")

        # Load tokenizer and model
        model = DDP(self.model, device_ids=[rank], output_device=rank)
        local_embeddings = {}
        for name, doc in doc_chunks.items():
            inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True).to(
                device
            )
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            local_embeddings[name] = embeddings.cpu()  # Move back to CPU

        queue.put(local_embeddings)
        # Synchronize processes & gather results
        dist.barrier()

        # Cleanup
        dist.destroy_process_group()


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
