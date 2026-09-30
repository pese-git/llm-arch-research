import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from llm.training.trainer import Trainer

# Синтетический небольшой датасет для автогрессивной LM задачи
class ToyLMDataset(Dataset):
    def __init__(self, num_samples=16, seq_len=8, vocab_size=16):
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # labels == input_ids (identity task)
        return {"input_ids": self.data[idx], "labels": self.data[idx]}

# Простая dummy-модель — 1 слой linear over vocab
class TinyModel(nn.Module):
    def __init__(self, vocab_size=16, seq_len=8):
        super().__init__()
        self.linear = nn.Linear(seq_len, vocab_size)
    def forward(self, x):
        # logits: (batch, seq_len, vocab_size)
        # Для простоты делаем транспонирование
        return self.linear(x.float()).unsqueeze(1).expand(-1, x.shape[1], -1)

def test_train_runs_without_errors():
    train_data = ToyLMDataset(num_samples=16, seq_len=8, vocab_size=16)
    model = TinyModel(vocab_size=16, seq_len=8)
    trainer = Trainer(model, train_data, lr=1e-3, batch_size=4, num_epochs=1, warmup_steps=2)
    trainer.train()

def test_trainer_evaluate_runs():
    train_data = ToyLMDataset(num_samples=8)
    val_data = ToyLMDataset(num_samples=8)
    model = TinyModel()
    trainer = Trainer(model, train_data, val_data, lr=1e-3, batch_size=4, num_epochs=1, warmup_steps=2)
    trainer.train()
    trainer.evaluate()

def test_trainer_tuple_output():
    # Модель, возвращающая кортеж (logits, extra)
    class TupleModel(nn.Module):
        def __init__(self, vocab_size=16, seq_len=8):
            super().__init__()
            self.linear = nn.Linear(seq_len, vocab_size)
        def forward(self, x):
            logits = self.linear(x.float()).unsqueeze(1).expand(-1, x.shape[1], -1)
            extra = torch.zeros(1)
            return logits, extra

    train_data = ToyLMDataset(num_samples=8)
    model = TupleModel()
    trainer = Trainer(model, train_data, lr=1e-3, batch_size=2, num_epochs=1, warmup_steps=1)
    trainer.train()

def test_trainer_loss_decreases():
    train_data = ToyLMDataset(num_samples=32, seq_len=8, vocab_size=8)
    model = TinyModel(vocab_size=8, seq_len=8)
    trainer = Trainer(model, train_data, lr=0.05, batch_size=8, num_epochs=2, warmup_steps=1)
    trainer.train()
    avg_losses = trainer.loss_history
    assert avg_losses[-1] <= avg_losses[0] or abs(avg_losses[-1] - avg_losses[0]) < 1e-3

class TupleModel(nn.Module):
    """Как модели llm: forward возвращает (logits, cache)."""

    def __init__(self, vocab_size=16, seq_len=8):
        super().__init__()
        self.linear = nn.Linear(seq_len, vocab_size)

    def forward(self, x):
        logits = self.linear(x.float()).unsqueeze(1).expand(-1, x.shape[1], -1)
        return logits, None


def test_trainer_evaluate_returns_average_loss():
    train_data = ToyLMDataset(num_samples=8)
    val_data = ToyLMDataset(num_samples=8)
    model = TinyModel()
    trainer = Trainer(model, train_data, val_data, lr=1e-3, batch_size=4, num_epochs=1, warmup_steps=2)

    loss = trainer.evaluate()

    assert isinstance(loss, float) and loss > 0
    assert not model.training  # evaluate переводит модель в eval()


def test_trainer_evaluate_tuple_output():
    """Валидация с моделью, которая, как модели llm, возвращает кортеж."""
    torch.manual_seed(0)
    train_data = ToyLMDataset(num_samples=8)
    val_data = ToyLMDataset(num_samples=8)
    tuple_model = TupleModel()
    tensor_model = TinyModel()
    tensor_model.load_state_dict(tuple_model.state_dict())

    tuple_loss = Trainer(tuple_model, train_data, val_data, batch_size=4).evaluate()
    tensor_loss = Trainer(tensor_model, train_data, val_data, batch_size=4).evaluate()

    assert tuple_loss == pytest.approx(tensor_loss)


def test_trainer_adds_auxiliary_loss():
    """Вспомогательный loss модели (load-balancing MoE) прибавляется к LM loss при обучении."""

    class ModelWithAuxLoss(TinyModel):
        def __init__(self):
            super().__init__()
            self.aux_calls = 0

        def auxiliary_loss(self):
            self.aux_calls += 1
            return torch.tensor(100.0)

    torch.manual_seed(0)
    train_data = ToyLMDataset()
    model = ModelWithAuxLoss()
    trainer = Trainer(model, train_data, lr=1e-3, batch_size=4, num_epochs=1, warmup_steps=1)
    trainer.train()

    assert model.aux_calls == len(trainer.train_loader)
    assert trainer.loss_history[0] > 100  # LM loss + 100


def test_trainer_models_without_auxiliary_loss():
    """Модели без auxiliary_loss (и с None) обучаются как раньше."""
    torch.manual_seed(0)
    trainer = Trainer(TinyModel(), ToyLMDataset(), lr=1e-3, batch_size=4, num_epochs=1, warmup_steps=1)
    trainer.train()
    assert trainer.loss_history[0] < 100


# --- Паддинг в loss (бэклог, пункт 57) ---

from llm.datasets.text_dataset import TextDataset
from llm.models.gpt import GPT
from llm.models.mixtral import Mixtral


class CharTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(c) - ord("a") + 1 for c in text]


GPT_CONFIG = {"vocab_size": 30, "embed_dim": 16, "num_heads": 2, "num_layers": 1,
              "max_position_embeddings": 16, "dropout": 0.0}
TEXTS = ["abcde", "fghijklm", "nop"]


def test_compute_lm_loss_ignores_padding():
    """Loss — среднее только по позициям с меткой, отличной от -100."""
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7)
    labels = torch.tensor([[3, 1, 4, -100, -100]])
    trainer = Trainer(TinyModel(), ToyLMDataset(), batch_size=1)
    expected = F.cross_entropy(logits[0, :2], labels[0, 1:3])
    assert trainer.compute_lm_loss(logits, labels).item() == pytest.approx(expected.item())


def test_compute_lm_loss_no_targets_is_zero_not_nan():
    """Батч из одного паддинга: loss 0 с графом, а не NaN, который испортил бы веса."""
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.full((2, 4), -100, dtype=torch.long)
    trainer = Trainer(TinyModel(), ToyLMDataset(), batch_size=1)
    loss = trainer.compute_lm_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_trainer_passes_attention_mask_to_model():
    """attention_mask из батча доходит до модели."""
    seen = []

    class MaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(30, 30)

        def forward(self, x, attention_mask=None):
            seen.append(attention_mask)
            return self.emb(x)

    dataset = TextDataset(TEXTS, CharTokenizer(), block_size=10)
    Trainer(MaskModel(), dataset, batch_size=3, num_epochs=1, warmup_steps=0).train()
    # Батч перемешан: сверяем число настоящих токенов в строках
    assert sorted(seen[0].sum(dim=1).tolist()) == sorted(len(t) for t in TEXTS)


def test_validation_loss_does_not_depend_on_padding_length():
    """Loss на реальной модели не зависит от block_size: паддинг не входит ни в loss,
    ни (благодаря causal-маске и attention_mask) в выход настоящих токенов."""
    torch.manual_seed(0)
    model = GPT(GPT_CONFIG)
    losses = []
    for block_size in (8, 16):
        dataset = TextDataset(TEXTS, CharTokenizer(), block_size=block_size)
        losses.append(Trainer(model, dataset, dataset, batch_size=3).evaluate())
    assert losses[0] == pytest.approx(losses[1], abs=1e-5)


def test_trainer_trains_real_model_with_padding():
    """Обучение GPT на примерах с паддингом: loss конечен и падает."""
    torch.manual_seed(0)
    dataset = TextDataset(TEXTS * 4, CharTokenizer(), block_size=12)
    trainer = Trainer(GPT(GPT_CONFIG), dataset, lr=1e-2, batch_size=4, num_epochs=3, warmup_steps=1)
    trainer.train()
    assert all(torch.isfinite(torch.tensor(trainer.loss_history)))
    assert trainer.loss_history[-1] < trainer.loss_history[0]


def test_mixtral_aux_loss_ignores_padding_from_trainer():
    """Trainer передаёт attention_mask, и Mixtral не учитывает паддинг в load-balancing loss."""
    torch.manual_seed(0)
    config = {"vocab_size": 30, "embed_dim": 16, "num_q_heads": 2, "num_kv_heads": 1,
              "num_layers": 1, "max_position_embeddings": 16, "dropout": 0.0,
              "num_experts": 4, "top_k_experts": 2, "router_aux_loss_coef": 0.01}
    model = Mixtral(config)
    dataset = TextDataset(TEXTS, CharTokenizer(), block_size=12)
    batch = {k: torch.stack([dataset[i][k] for i in range(3)]) for k in dataset[0]}
    trainer = Trainer(model, dataset, batch_size=3)
    trainer._forward(batch)
    assert model._aux_token_mask is not None
    assert int(model._aux_token_mask.sum()) == sum(len(t) for t in TEXTS)


# --- Warmup (бэклог, пункт 60) ---

import warnings


def test_warmup_default_is_100_steps():
    trainer = Trainer(TinyModel(), ToyLMDataset(), batch_size=4)
    assert trainer.warmup_steps == 100 and trainer.warmup_ratio is None
    assert trainer.num_warmup_steps(1000) == 100


@pytest.mark.parametrize("ratio, total, expected", [(0.1, 18, 2), (0.1, 20, 2), (0.05, 1000, 50), (0.0, 18, 0), (1.0, 18, 18)])
def test_warmup_ratio_is_fraction_of_steps(ratio, total, expected):
    """warmup_ratio → ceil(N_steps · ratio), как в HuggingFace TrainingArguments."""
    trainer = Trainer(TinyModel(), ToyLMDataset(), batch_size=4, warmup_ratio=ratio)
    assert trainer.warmup_steps is None
    assert trainer.num_warmup_steps(total) == expected


@pytest.mark.parametrize("kwargs", [
    {"warmup_steps": 5, "warmup_ratio": 0.1},
    {"warmup_steps": -1},
    {"warmup_ratio": -0.1},
    {"warmup_ratio": 1.5},
])
def test_warmup_invalid_args(kwargs):
    with pytest.raises(ValueError):
        Trainer(TinyModel(), ToyLMDataset(), batch_size=4, **kwargs)


def test_warmup_ratio_reaches_peak_lr():
    """С warmup_ratio learning rate доходит до заданного даже при коротком обучении."""
    torch.manual_seed(0)
    # 16 примеров, batch 4, 3 эпохи — 12 шагов; warmup ceil(1.2) = 2
    trainer = Trainer(TinyModel(), ToyLMDataset(), lr=1e-3, batch_size=4, num_epochs=3, warmup_ratio=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # предупреждения о длинном warmup нет
        trainer.train()
    # Множитель learning rate на шагах 0 … 11: warmup 0, 0.5, затем спад от 1
    factors = [trainer.scheduler.lr_lambdas[0](k) for k in range(12)]
    assert factors[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert max(factors) == pytest.approx(1.0)


def test_warmup_longer_than_training_warns():
    """warmup_steps ≥ числа шагов — предупреждение: learning rate не дойдёт до заданного."""
    # 16 примеров, batch 4, 1 эпоха — 4 шага
    trainer = Trainer(TinyModel(), ToyLMDataset(), batch_size=4, num_epochs=1, warmup_steps=50)
    with pytest.warns(UserWarning, match="warmup_steps = 50"):
        trainer.train()
