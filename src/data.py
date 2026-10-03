"""data.py: Dataset PyTorch untuk teks ulasan dan rating (Subbab 3.9). Backbone tunggal IndoBERT."""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src import config

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    return _tokenizer


class ReviewDataset(Dataset):
    """Menerima label mentah 1 sampai 5 dan mengonversinya ke 0 sampai 4 di satu tempat."""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [int(l) - 1 for l in labels]
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), add_special_tokens=True, max_length=config.MAX_LEN,
            padding="max_length", truncation=True, return_token_type_ids=False,
            return_attention_mask=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].flatten(),
                "attention_mask": enc["attention_mask"].flatten(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
