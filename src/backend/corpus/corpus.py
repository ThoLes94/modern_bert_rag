from pathlib import Path
from typing import Dict, Iterator, List

from torch.utils.data import DataLoader, IterableDataset

# class Corpus:
#     bright_biology =


class DatasetWrapper(IterableDataset[Dict[str, str]]):
    def __init__(self, path_to_files: str, chunk_size: int = 2048) -> None:
        self.root = path_to_files
        self.find_all_files()
        self.chunk_size = chunk_size

    def find_all_files(self) -> None:
        self.list_files: List[Path] = list(Path(self.root).rglob("*.txt"))
        assert len(self.list_files), "No documents found"

    def _read_file_in_chunks(self, file_path: Path) -> Iterator[Dict[str, str]]:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        if len(content) <= self.chunk_size:
            yield {"id": str(file_path).replace("/", "_"), "content": content}
        else:
            for i in range(0, len(content), self.chunk_size):
                chunk = content[i : i + self.chunk_size]
                yield {
                    "id": str(file_path).replace("/", "_") + f"_{i // self.chunk_size}",
                    "content": chunk,
                }

    def __iter__(self) -> Iterator[Dict[str, str]]:
        for file_path in self.list_files:
            yield from self._read_file_in_chunks(file_path)


if __name__ == "__main__":
    corpus = DatasetWrapper("data/corpus/docs.mistral.ai")

    dataloader = DataLoader(corpus, batch_size=2)
    for k in dataloader:
        print(k)
        break

    # print(len(dataloader) * 16)
