# from chainlit.utils import mount_chainlit
import json
import logging
import time
from typing import Any, Dict

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.backend.corpus.corpus import DatasetWrapper
from src.backend.models.encoder import BertHFPath

logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure logging to only write to a file
logging.basicConfig(
    filename="query_logs.log",  # Log file path
    level=logging.INFO,  # Set log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filemode="w",  # Overwrite the file on each run (use "a" to append)
)

device = "cuda" if torch.cuda.is_available() else "mps"


def benchmark_on_corpus(
    encoder_path: BertHFPath,
    batch_size: int = 10,
    chunk_size: int = 4096,
    num_pass: int = 1,
    use_random_chunk_size: bool = False,
) -> Dict[str, Any]:
    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(encoder_path.value, trust_remote_code=True)
    model.to(device, torch.float16)

    corpus = DatasetWrapper(
        "data/corpus/docs.mistral.ai",
    )
    dataloader = corpus.get_dataloader(
        batch_size=batch_size, tokenizer_path=encoder_path, chunk_size=chunk_size, tokenize=True
    )

    evaluation_times = []
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            for docs in dataloader:
                for i in range(2):
                    inputs = {
                        k: v.to(device) for k, v in docs.items() if k not in ["content", "id"]
                    }
                    model(**inputs)
                break

    num_tokken = 0
    token_counts = []

    start_time = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            for i in tqdm(range(num_pass)):
                for docs in tqdm(dataloader):
                    inputs = {
                        k: v.to(device) for k, v in docs.items() if k not in ["content", "id"]
                    }
                    num_tokken += inputs["input_ids"].numel()
                    start_evaluation = time.perf_counter()
                    model(**inputs)
                    evaluation_time = time.perf_counter() - start_evaluation
                    evaluation_times.append(evaluation_time)
                    token_counts.append(inputs["input_ids"].numel())

    total_time = time.perf_counter() - start_time

    mean_time = np.mean(evaluation_times)
    var_time = np.var(evaluation_times)

    tokens_per_sec = np.sum(np.array(token_counts) / 1e3) / total_time

    results = {
        "exp": {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "encoder": encoder_path.value,
            "use_random_chunk_size": use_random_chunk_size,
        },
        "total_time": str(total_time),
        "mean_time": str(sum(evaluation_times) / len(evaluation_times)),
        "num_tokken": num_tokken,
        "var": var_time,
        "time/tokken": mean_time / np.mean(num_tokken),
        "tokken/sec": tokens_per_sec,
    }

    logging.info(json.dumps(results))

    return results


if __name__ == "__main__":
    print(benchmark_on_corpus(BertHFPath.gte_base, batch_size=16, chunk_size=256))
