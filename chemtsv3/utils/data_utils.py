import random
from collections.abc import Callable

def load_texts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if line.strip()]

def train_test_split(examples: list[str], test_size: float = 0.1) -> dict[str, list[str]]:
    examples = list(examples)
    if not examples:
        raise ValueError("Text dataset is empty.")
    if test_size is None:
        test_size = 0.1
    random.shuffle(examples)
    n_test = max(1, int(len(examples) * test_size))
    return {
        "train": examples[n_test:],
        "test": examples[:n_test],
    }

def map_text_splits(splits: dict[str, list[str]], map_fn: Callable[[str], dict]) -> dict[str, list[dict]]:
    return {split: [map_fn(example) for example in examples] for split, examples in splits.items()}
