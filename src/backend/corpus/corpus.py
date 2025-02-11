import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer


class DatasetWrapper(IterableDataset[Dict[str, str]]):
    def __init__(
        self,
        path_to_files: str,
        chunk_size: Optional[int] = 2048,
        tokenizer_path: str = "lightonai/modernbert-embed-large",
        use_random_chunk_size: bool = False,
    ) -> None:
        self.root = path_to_files
        self.find_all_files()
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.use_random_chunk_size = use_random_chunk_size

    def find_all_files(self) -> None:
        self.list_files: List[Path] = list(Path(self.root).rglob("*.txt"))
        assert len(self.list_files), "No documents found"

    def _read_file_in_chunks(self, file_path: Path) -> Iterator[Dict[str, str]]:
        with file_path.open() as f:
            content = f.read()

        tokens = self.tokenizer(content, truncation=False, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)

        assert self.chunk_size is not None
        index = 0
        while index < len(tokens):
            chunk_size = (
                self.chunk_size
                if not self.use_random_chunk_size
                else random.randint(1, self.chunk_size)
            )
            chunk_tokens = tokens[index : index + chunk_size]
            chunk_text = self.tokenizer.decode(
                chunk_tokens.squeeze(0).tolist(), skip_special_tokens=True
            )
            index += chunk_size

            yield {
                "id": f"{file_path.stem}_{index // self.chunk_size}",
                "content": chunk_text,
            }

    def __iter__(self) -> Iterator[Dict[str, str]]:
        for file_path in self.list_files:
            yield from self._read_file_in_chunks(file_path)

    def tokenization(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(example["content"], return_tensors="pt", padding=True)


if __name__ == "__main__":
    corpus = DatasetWrapper(
        "data/corpus/docs.mistral.ai",
        chunk_size=8192,  # tokenizer="Alibaba-NLP/gte-large-en-v1.5"
    )
    dataset = Dataset.from_generator(corpus.__iter__)
    dataset = dataset.map(corpus.tokenization, batched=True)
    dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=10)

    for k in dataloader:
        print((k["input_ids"]))
        break
