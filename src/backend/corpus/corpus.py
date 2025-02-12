import random
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional, cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.backend.models.encoder import BertHFPath


class DatasetWrapper:
    def __init__(
        self,
        path_to_files: str,
    ) -> None:
        self.root = path_to_files
        self.find_all_files()

    def find_all_files(self) -> None:
        self.list_files: List[Path] = list(Path(self.root).rglob("*.txt"))
        assert len(self.list_files), "No documents found"

    def _read_file_in_chunks(
        self,
        file_path: Path,
        tokenizer: AutoTokenizer,
        chunk_size: int,
        use_random_chunk_size: bool = False,
    ) -> Iterator[Dict[str, str]]:
        with file_path.open() as f:
            content = f.read()

        tokens = tokenizer(content, truncation=False, return_tensors="pt")["input_ids"].squeeze(0)

        assert chunk_size is not None
        index = 0
        while index < len(tokens):
            chunk_size = chunk_size if not use_random_chunk_size else random.randint(1, chunk_size)
            chunk_tokens = tokens[index : index + chunk_size]
            chunk_text = tokenizer.decode(
                chunk_tokens.squeeze(0).tolist(), skip_special_tokens=True
            )
            index += chunk_size

            yield {
                "id": f"{str(file_path).replace('/', '_')}_{index}_{chunk_size}",
                "content": chunk_text,
            }

    def get_iterator(
        self,
        tokenizer: AutoTokenizer,
        chunk_size: int = 2048,
        use_random_chunk_size: bool = False,
    ) -> Iterator[Dict[str, str]]:
        for file_path in self.list_files:
            yield from self._read_file_in_chunks(
                file_path, tokenizer, chunk_size, use_random_chunk_size
            )

    def get_dataloader(
        self,
        batch_size: int = 10,
        tokenizer_path: BertHFPath = BertHFPath.modern_bert_base_embed,
        tokenize: bool = False,
        chunk_size: int = 2048,
        use_random_chunk_size: bool = False,
    ) -> DataLoader[Dict[str, torch.Tensor]]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.value)

        dataset = Dataset.from_generator(
            partial(
                self.get_iterator,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                use_random_chunk_size=use_random_chunk_size,
            )
        )
        if tokenize:
            dataset = dataset.map(partial(self.tokenization, tokenizer=tokenizer), batched=True)
            columns_of_interest = {"input_ids", "attention_mask", "token_type_ids"}.intersection(
                dataset.column_names
            )

            dataset.set_format("pt", columns=columns_of_interest, output_all_columns=False)

        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader

    def tokenization(
        self,
        example: Dict[str, torch.Tensor],
        tokenizer: BertHFPath = BertHFPath.modern_bert_base_embed,
    ) -> Dict[str, torch.Tensor]:

        return cast(
            Dict[str, torch.Tensor],
            tokenizer(example["content"], return_tensors="pt", padding=True),
        )


if __name__ == "__main__":
    corpus = DatasetWrapper(
        "data/corpus/docs.mistral.ai",
    )

    dataloader = corpus.get_dataloader(chunk_size=512, tokenize=True)
    for k in dataloader:
        print((k.keys()))
        break
