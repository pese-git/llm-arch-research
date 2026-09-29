import pytest
import torch
import torch.nn as nn
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
