from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BERTTextEncoder(nn.Module):
    """Hugging Face BERT/DistilBERT encoder for text sequence and pooled token outputs."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, texts: List[str] | str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hidden states H_text and pooled [CLS] embedding t."""
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids 2D, got shape {input_ids.shape}")

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        H_text = outputs.last_hidden_state
        t = H_text[:, 0, :]
        t = self.dropout(t)

        assert H_text.ndim == 3, f"H_text must be [B, L, d], got {H_text.shape}"
        assert t.ndim == 2, f"t must be [B, d], got {t.shape}"
        assert H_text.size(-1) == self.hidden_dim, (
            f"Unexpected hidden size: {H_text.size(-1)} != {self.hidden_dim}"
        )

        return H_text, t


if __name__ == "__main__":
    encoder = BERTTextEncoder()
    texts = [
        "calm piano melody with a soft warm texture",
        "energetic electronic drums and bright synths",
    ]
    H_text, t = encoder(texts)
    print(f"H_text: {H_text.shape}")
    print(f"t: {t.shape}")
