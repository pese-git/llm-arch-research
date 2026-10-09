"""
Чекпоинт обучения: один файл, надмножество формата `BaseModel.save`.

Файл `torch.save` содержит те же ключи, что пишет `BaseModel.save` —
`model_class`, `config`, `state_dict` — поэтому `BaseModel.load(path)` читает его
как обычную модель, а под ключом `trainer` лежит всё, что нужно продолжить обучение
с того же шага: состояние оптимизатора и планировщика, номер шага, история и
генератор случайных чисел. Все значения — тензоры, числа, строки, списки и словари:
файл читается с `weights_only=True` (ADR-003 в docs/dev/decisions.md).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

FORMAT_VERSION = 1


@dataclass
class TrainState:
    """
    Изменяемое состояние обучения, которое попадает в чекпоинт.

    step — число сделанных шагов оптимизатора; epoch — номер текущей эпохи (с нуля);
    step_in_epoch — сколько шагов сделано внутри текущей эпохи (чтобы после resume
    пропустить уже пройденные батчи); best_val_loss — лучший валидационный loss;
    loss_history — средний train loss по завершённым эпохам; log — записи
    {"step", "epoch", "lr", "train_loss", "val_loss"}.
    """

    step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0
    best_val_loss: Optional[float] = None
    loss_history: List[float] = field(default_factory=list)
    log: List[Dict[str, Any]] = field(default_factory=list)


def rng_state() -> Dict[str, Any]:
    """Состояние генераторов случайных чисел PyTorch: CPU и, если есть, все CUDA-устройства."""
    state: Dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainState,
    args: Dict[str, Any],
) -> None:
    """
    Пишет чекпоинт обучения в path (папку не создаёт — это делает вызывающий).

    Args:
        model: модель; `config` берётся из `model.config`, если есть (как в BaseModel.save).
        optimizer, scheduler: их `state_dict()`; scheduler может быть None до начала обучения.
        state: TrainState.
        args: аргументы Trainer, от которых зависит расписание; проверяются при resume.
    """
    config = getattr(model, "config", None)
    checkpoint = {
        "model_class": type(model).__name__,
        "config": dict(config) if config is not None else None,
        "state_dict": model.state_dict(),
        "trainer": {
            "format_version": FORMAT_VERSION,
            "state": asdict(state),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": rng_state(),
            "args": dict(args),
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Читает чекпоинт обучения с `weights_only=True` и проверяет его формат.

    Raises:
        ValueError: файл не содержит секции `trainer` (например, это файл `BaseModel.save`)
            или её версия формата неизвестна.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "trainer" not in checkpoint:
        raise ValueError(
            f"{path}: это не чекпоинт обучения (нет секции trainer); "
            "файл BaseModel.save загружайте через BaseModel.load"
        )
    version = checkpoint["trainer"].get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: версия формата чекпоинта {version}, поддерживается {FORMAT_VERSION}"
        )
    return checkpoint
