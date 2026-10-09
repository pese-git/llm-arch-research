"""
Trainer: устройство, обучение по шагам, валидация по интервалу, чекпоинты, resume, лог, seed.
"""

import json

import pytest
import torch

from llm.core.base_model import BaseModel
from llm.datasets.token_block_dataset import TokenBlockDataset
from llm.models.gpt import GPT
from llm.training.checkpoint import load_checkpoint
from llm.training.trainer import Trainer, resolve_device

VOCAB = 30
GPT_CONFIG = {"vocab_size": VOCAB, "embed_dim": 16, "num_heads": 2, "num_layers": 1,
              "max_position_embeddings": 8, "dropout": 0.0}


def dataset(num_tokens=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return TokenBlockDataset(torch.randint(0, VOCAB, (num_tokens,), generator=g).tolist(), block_size=8)


def make_trainer(model=None, **kwargs):
    kwargs.setdefault("batch_size", 4)      # 96 токенов / 8 = 12 блоков = 3 батча в эпохе
    kwargs.setdefault("warmup_steps", 2)
    kwargs.setdefault("lr", 1e-2)
    return Trainer(model or GPT(GPT_CONFIG), dataset(), dataset(64, seed=1), **kwargs)


def weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_same_weights(a, b, atol=1e-6):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol), k


# ------------------------------------------------------------------ устройство

def test_default_device_ignores_mps(monkeypatch):
    """None — cuda или cpu, как раньше: ноутбуки и иллюстрации подают CPU-тензоры."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device(None).type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None).type == "cuda"


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_explicit_device_and_errors(monkeypatch):
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
    with pytest.raises(ValueError, match="Неизвестное устройство"):
        resolve_device("xla")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="недоступно"):
        resolve_device("cuda")


def test_trainer_moves_model_to_requested_device():
    trainer = make_trainer(device="cpu")
    assert trainer.device.type == "cpu"
    assert next(trainer.model.parameters()).device.type == "cpu"


# ------------------------------------------------------------------ аргументы

@pytest.mark.parametrize("kwargs", [
    {"max_steps": 0},
    {"eval_interval": 0},
    {"save_interval": 0, "checkpoint_dir": "x"},
    {"eval_batches": 0},
    {"save_interval": 2},                       # без checkpoint_dir
])
def test_invalid_args(kwargs):
    with pytest.raises(ValueError):
        make_trainer(device="cpu", **kwargs)


def test_eval_interval_requires_val_dataset():
    with pytest.raises(ValueError, match="val_dataset"):
        Trainer(GPT(GPT_CONFIG), dataset(), eval_interval=1, device="cpu")


# ------------------------------------------------------------------ шаги

def test_max_steps_runs_exactly_that_many_steps():
    """7 шагов при 3 батчах в эпохе: третья эпоха прервана после первого батча, lr дошёл до 0."""
    trainer = make_trainer(device="cpu", max_steps=7)
    trainer.train()
    assert trainer.state.step == 7
    assert trainer.scheduler.last_epoch == 7
    assert trainer.state.epoch == 2 and trainer.state.step_in_epoch == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    assert len(trainer.loss_history) == 3  # две полные эпохи и начатая третья


def test_epoch_mode_unchanged():
    """Без max_steps — по эпохам, как раньше: loss_history по эпохе, валидация в конце каждой."""
    trainer = make_trainer(device="cpu", num_epochs=2)
    trainer.train()
    assert trainer.state.step == 6 and trainer.state.epoch == 2
    assert len(trainer.loss_history) == 2
    assert [r["step"] for r in trainer.state.log] == [3, 6]
    assert all("val_loss" in r for r in trainer.state.log)
    assert not trainer.model.training  # после финальной валидации — eval, как раньше


def test_eval_interval_and_eval_batches():
    """Валидация каждые 2 шага и в конце; eval_batches ограничивает число батчей."""
    calls = []
    trainer = make_trainer(device="cpu", max_steps=5, eval_interval=2, eval_batches=1)
    original = trainer._forward

    def counting_forward(batch):
        if not trainer.model.training:
            calls.append(batch["input_ids"].shape[0])
        return original(batch)

    trainer._forward = counting_forward
    trainer.train()
    assert [r["step"] for r in trainer.state.log] == [2, 4, 5]
    assert len(calls) == 3  # по одному батчу на каждую из трёх валидаций
    assert all(r["train_loss"] > 0 and r["lr"] >= 0 for r in trainer.state.log)


# ------------------------------------------------------------------ чекпоинты

def test_last_checkpoint_loads_as_model(tmp_path):
    trainer = make_trainer(device="cpu", max_steps=4, checkpoint_dir=str(tmp_path), save_interval=2)
    trainer.train()
    path = tmp_path / "last.pt"
    assert path.exists()
    restored = GPT.load(str(path))
    x = torch.randint(0, VOCAB, (2, 8))
    trainer.model.eval()
    assert torch.allclose(trainer.model(x)[0], restored(x)[0])
    assert isinstance(restored, BaseModel)
    checkpoint = load_checkpoint(str(path))
    assert checkpoint["trainer"]["state"]["step"] == 4
    assert checkpoint["trainer"]["args"]["max_steps"] == 4


def test_resume_is_equivalent_to_uninterrupted_training(tmp_path):
    """6 шагов подряд == 3 шага, save, новый Trainer, resume, 3 шага (seed, dropout 0, CPU)."""
    torch.manual_seed(0)
    straight = make_trainer(GPT(GPT_CONFIG), device="cpu", max_steps=6, seed=123)
    straight.train()

    torch.manual_seed(0)
    first = make_trainer(GPT(GPT_CONFIG), device="cpu", max_steps=6, seed=123,
                         checkpoint_dir=str(tmp_path), save_interval=3)
    first.train_loader_len = len(first.train_loader)
    # Прерываем после 3 шагов: обучаем копию с max_steps=3 нельзя (другие args), поэтому
    # останавливаем через исключение из _train_step
    steps = {"n": 0}
    original_step = first._train_step

    def stop_after_three(batch):
        if steps["n"] == 3:
            raise KeyboardInterrupt
        steps["n"] += 1
        return original_step(batch)

    first._train_step = stop_after_three
    with pytest.raises(KeyboardInterrupt):
        first.train()
    assert (tmp_path / "last.pt").exists() and first.state.step == 3

    second = make_trainer(GPT(GPT_CONFIG), device="cpu", max_steps=6, seed=123,
                          checkpoint_dir=str(tmp_path))
    second.resume(str(tmp_path / "last.pt"))
    assert second.state.step == 3 and second.state.step_in_epoch == 0 and second.state.epoch == 1
    second.train()
    assert second.state.step == 6
    assert_same_weights(weights(straight.model), weights(second.model))
    # Лог и история продолжены, а не начаты заново
    assert [r["step"] for r in second.state.log] == [r["step"] for r in straight.state.log]
    assert second.loss_history == pytest.approx(straight.loss_history)


def test_resume_mid_epoch_skips_seen_batches(tmp_path):
    """Чекпоинт посреди эпохи: после resume пропускаются уже пройденные батчи той же перестановки."""
    torch.manual_seed(0)
    straight = make_trainer(GPT(GPT_CONFIG), device="cpu", max_steps=5, seed=7)
    straight.train()

    torch.manual_seed(0)
    first = make_trainer(GPT(GPT_CONFIG), device="cpu", max_steps=5, seed=7,
                         checkpoint_dir=str(tmp_path), save_interval=4)
    original_step = first._train_step
    done = {"n": 0}

    def stop_after_four(batch):
        if done["n"] == 4:
            raise KeyboardInterrupt
        done["n"] += 1
        return original_step(batch)

    first._train_step = stop_after_four
    with pytest.raises(KeyboardInterrupt):
        first.train()

    torch.manual_seed(0)
    second = make_trainer(GPT(GPT_CONFIG), device="cpu", max_steps=5, seed=7)
    second.resume(str(tmp_path / "last.pt"))
    assert second.state.epoch == 1 and second.state.step_in_epoch == 1
    second.train()
    assert_same_weights(weights(straight.model), weights(second.model))


def test_resume_rejects_other_model_or_args(tmp_path):
    trainer = make_trainer(device="cpu", max_steps=2, checkpoint_dir=str(tmp_path))
    trainer.train()
    path = str(tmp_path / "last.pt")
    with pytest.raises(ValueError, match="max_steps"):
        make_trainer(device="cpu", max_steps=3).resume(path)
    with pytest.raises(ValueError, match="lr"):
        make_trainer(device="cpu", max_steps=2, lr=5e-3).resume(path)

    class Other(GPT):
        pass

    with pytest.raises(ValueError, match="Other"):
        make_trainer(Other(GPT_CONFIG), device="cpu", max_steps=2).resume(path)


def test_resume_rejects_plain_model_file(tmp_path):
    model = GPT(GPT_CONFIG)
    model.save(str(tmp_path / "model.pt"))
    with pytest.raises(ValueError, match="не чекпоинт обучения"):
        make_trainer(device="cpu").resume(str(tmp_path / "model.pt"))


def test_best_checkpoint_saved_only_on_improvement(tmp_path, monkeypatch):
    trainer = make_trainer(device="cpu", max_steps=3, eval_interval=1, checkpoint_dir=str(tmp_path))
    losses = iter([2.0, 1.0, 1.5])
    monkeypatch.setattr(trainer, "evaluate", lambda: next(losses))
    saved = []
    original = trainer.save_checkpoint
    monkeypatch.setattr(trainer, "save_checkpoint",
                        lambda path: (saved.append((trainer.state.step, path.split("/")[-1])), original(path)))
    trainer.train()
    assert [s for s in saved if s[1] == "best.pt"] == [(1, "best.pt"), (2, "best.pt")]
    assert trainer.state.best_val_loss == 1.0
    assert load_checkpoint(str(tmp_path / "best.pt"))["trainer"]["state"]["step"] == 2


def test_keep_best_false_writes_no_best(tmp_path):
    trainer = make_trainer(device="cpu", max_steps=2, eval_interval=1, checkpoint_dir=str(tmp_path),
                           keep_best=False)
    trainer.train()
    assert (tmp_path / "last.pt").exists() and not (tmp_path / "best.pt").exists()


def test_log_path_written(tmp_path):
    log_path = tmp_path / "logs" / "log.json"
    trainer = make_trainer(device="cpu", max_steps=4, eval_interval=2, log_path=str(log_path))
    trainer.train()
    records = json.loads(log_path.read_text())
    assert [r["step"] for r in records] == [2, 4]
    assert set(records[0]) == {"step", "epoch", "lr", "train_loss", "val_loss"}


def test_seed_makes_batch_order_reproducible():
    def order(seed):
        trainer = make_trainer(device="cpu", seed=seed)
        trainer._generator.manual_seed(trainer.seed + 0)
        return [b["input_ids"][0, 0].item() for b in trainer.train_loader]

    assert order(5) == order(5)
    assert order(5) != order(6)
