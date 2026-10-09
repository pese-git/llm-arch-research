"""
llm.evaluation: lm_loss и perplexity — число руками, равенство Trainer.evaluate,
паддинг через attention_mask, max_batches.
"""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from llm.datasets.text_dataset import TextDataset
from llm.datasets.token_block_dataset import TokenBlockDataset
from llm.evaluation import lm_loss, perplexity
from llm.models.gpt import GPT
from llm.training.trainer import Trainer

VOCAB = 30
GPT_CONFIG = {"vocab_size": VOCAB, "embed_dim": 16, "num_heads": 2, "num_layers": 1,
              "max_position_embeddings": 16, "dropout": 0.0}


class UniformModel(nn.Module):
    """Нулевые логиты для любого входа: модель, которая ничего не знает."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention_mask=None):
        return torch.zeros(*x.shape, VOCAB) + self.dummy


class CharTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(c) - ord("a") + 1 for c in text]


def test_uniform_model_perplexity_is_vocab_size():
    """Равномерная модель: loss = ln V, perplexity = V — независимо от данных."""
    torch.manual_seed(0)
    dataset = TokenBlockDataset(torch.randint(0, VOCAB, (64,)).tolist(), block_size=8)
    loader = DataLoader(dataset, batch_size=4)
    assert lm_loss(UniformModel(), loader) == pytest.approx(math.log(VOCAB))
    assert perplexity(UniformModel(), loader) == pytest.approx(VOCAB)


def test_perplexity_equals_exp_of_trainer_evaluate():
    """На одном наборе perplexity == exp(Trainer.evaluate()): одна функция loss, то же среднее."""
    torch.manual_seed(0)
    model = GPT(GPT_CONFIG)
    dataset = TextDataset(["abcde", "fghijklm", "nop", "qrstuv"], CharTokenizer(), block_size=8)
    trainer = Trainer(model, dataset, val_dataset=dataset, batch_size=2)
    val_loss = trainer.evaluate()
    loader = DataLoader(dataset, batch_size=2)
    assert lm_loss(model, loader, device=trainer.device) == pytest.approx(val_loss, rel=1e-6)
    assert perplexity(model, loader, device=trainer.device) == pytest.approx(math.exp(val_loss), rel=1e-6)


def test_padding_does_not_change_loss():
    """attention_mask из батча доходит до модели: паддинг справа не меняет loss."""
    torch.manual_seed(0)
    model = GPT(GPT_CONFIG)
    texts = ["abcde", "fgh"]
    short = DataLoader(TextDataset(texts, CharTokenizer(), block_size=6), batch_size=2)
    long = DataLoader(TextDataset(texts, CharTokenizer(), block_size=12), batch_size=2)
    assert lm_loss(model, short) == pytest.approx(lm_loss(model, long), rel=1e-5)


def test_max_batches_limits_and_empty_loader_raises():
    class Counting(UniformModel):
        calls = 0

        def forward(self, x, attention_mask=None):
            Counting.calls += 1
            return super().forward(x)

    dataset = TokenBlockDataset([i % VOCAB for i in range(80)], block_size=8)
    loader = DataLoader(dataset, batch_size=2)  # 5 батчей
    lm_loss(Counting(), loader, max_batches=2)
    assert Counting.calls == 2
    with pytest.raises(ValueError, match="батча"):
        lm_loss(UniformModel(), [])


def test_model_left_in_eval_mode():
    model = GPT(GPT_CONFIG).train()
    loader = DataLoader(TokenBlockDataset([i % VOCAB for i in range(32)], block_size=8), batch_size=2)
    lm_loss(model, loader)
    assert not model.training
