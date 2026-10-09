"""
Модуль для организации процесса обучения больших языковых моделей (LLM).

Научное и техническое обоснование
----------------------------------
Эффективное обучение современных трансформеров (GPT, LLaMA, Mistral и др.) опирается на принципы языкового моделирования (Language Modeling):
- Предсказание вероятности следующего токена на основе предыдущих.
- Использование функции потерь кросс-энтропии (cross-entropy) с маскированием паддингов.
- Циклы обратного распространения ошибки (backpropagation), оптимизационные алгоритмы (например, AdamW), управление шагом обучения (scheduler с warmup), обрезка градиентов (grad clipping).

Реализация объединяет лучшие практики обучения LLM, универсальный API к моделям, датасетам, оптимизаторам и lr-схемам.

Подробнее: Vaswani et al. "Attention is All You Need" (2017), Radford et al. "Language Models are Unsupervised Multitask Learners" (2019)

Пример использования
--------------------
>>> trainer = Trainer(model, train_dataset, val_dataset, lr=3e-4, batch_size=8, num_epochs=3, warmup_steps=100)
>>> trainer.train()

Обучение по шагам с чекпоинтами и продолжением:

>>> trainer = Trainer(model, train_dataset, val_dataset, max_steps=5000, eval_interval=500,
...                   checkpoint_dir="checkpoints/run", save_interval=500, log_path="run/log.json", seed=0)
>>> trainer.train()
>>> # позже, в новом процессе, с теми же аргументами:
>>> Trainer(model, train_dataset, val_dataset, max_steps=5000, ...).resume("checkpoints/run/last.pt").train()
"""
import json
import math
import os
import warnings
from typing import Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm.training.checkpoint import (
    TrainState,
    load_checkpoint,
    save_checkpoint,
    set_rng_state,
)
from llm.training.loss import causal_lm_loss
from llm.training.optimizer import get_optimizer
from llm.training.scheduler import get_linear_schedule_with_warmup

# Аргументы Trainer, от которых зависят число шагов и расписание learning rate:
# при resume чекпоинт с другими значениями отвергается
SCHEDULE_ARGS = ("lr", "batch_size", "num_epochs", "max_steps", "warmup_steps", "warmup_ratio")


def resolve_device(device: Union[str, torch.device, None]) -> torch.device:
    """
    Устройство для обучения.

    None — cuda, если доступна, иначе cpu (поведение по умолчанию: ноутбуки и
    иллюстрации учебника считают на CPU и подают модели CPU-тензоры). "auto" —
    первое доступное из cuda, mps, cpu: на Apple Silicon это MPS, в несколько раз
    быстрее CPU. Иначе — явное устройство.

    Raises:
        ValueError: неизвестный тип устройства или оно недоступно на этой машине.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device)
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if device.type not in available:
        raise ValueError(f"Неизвестное устройство {device}: ожидается cuda, mps или cpu")
    if not available[device.type]:
        raise ValueError(f"Устройство {device} недоступно на этой машине")
    return device


class Trainer:
    """
    Универсальный и расширяемый класс для обучения больших языковых моделей (Large Language Models, LLM).

    Поддерживаются архитектуры семейства GPT, LLaMA, Mistral и другие автогрессивные модели.
    Объединяет:
      - Тренировку по задаче языкового моделирования (Causal LM)
      - Cross-entropy loss с автоматическим сдвигом логитов/меток
      - Поддержку Grad Clipping, Scheduler, Validation
      - Обучение по эпохам или по числу шагов, валидацию и чекпоинты по интервалу,
        продолжение обучения с чекпоинта (`resume`)
      - Унифицированный даталоадер, выбор устройства (CUDA / CPU, с device="auto" — и MPS)

    Атрибуты
    --------
    model : torch.nn.Module
        Модель для обучения языковому моделированию
    train_loader : torch.utils.data.DataLoader
        Даталоадер обучающего набора
    val_loader : torch.utils.data.DataLoader или None
        Даталоадер валидационного набора (если задан)
    optimizer : torch.optim.Optimizer
        Оптимизатор параметров модели
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Планировщик learning rate (инициализируется в train)
    device : torch.device
        Устройство, куда помещается модель
    num_epochs : int
        Количество эпох обучения (если не задан max_steps)
    warmup_steps : int или None
        Число шагов warmup для scheduler (None, если задан warmup_ratio)
    warmup_ratio : float или None
        Доля warmup от общего числа шагов (None, если задан warmup_steps)
    state : TrainState
        Шаг, эпоха, лучший валидационный loss, история и лог; попадает в чекпоинт
    loss_history : list[float]
        Средний train loss по завершённым эпохам (то же, что state.loss_history)
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        lr=3e-4,
        batch_size=8,
        num_epochs=3,
        warmup_steps=None,
        warmup_ratio=None,
        *,
        device=None,
        max_steps=None,
        eval_interval=None,
        eval_batches=None,
        checkpoint_dir=None,
        save_interval=None,
        keep_best=True,
        log_path=None,
        seed=None,
    ):
        """
        Инициализация обучающего класса Trainer.

        Аргументы
        ---------
        model : torch.nn.Module
            Модель для обучения (например, GPT, LLaMA, Mistral).
        train_dataset : torch.utils.data.Dataset
            Обучающий датасет с полями input_ids и labels (и, если есть паддинг, attention_mask).
            Паддинг исключается из loss метками -100 (датасеты llm.datasets ставят их сами).
        val_dataset : torch.utils.data.Dataset, optional
            Валидационный датасет для контроля качества обучения.
        lr : float, default=3e-4
            Начальный шаг обучения.
        batch_size : int, default=8
            Размер обучающего мини-батча.
        num_epochs : int, default=3
            Количество эпох обучения. Игнорируется, если задан max_steps.
        warmup_steps : int, optional
            Количество шагов разогрева (warmup) learning rate. Если не задан ни он,
            ни warmup_ratio — 100.
        warmup_ratio : float, optional
            Warmup как доля от общего числа шагов, от 0 до 1: ceil(N_steps · warmup_ratio),
            как warmup_ratio в HuggingFace TrainingArguments. Удобнее warmup_steps, когда
            число шагов зависит от размера датасета. Нельзя вместе с warmup_steps.
        device : str | torch.device, optional
            Устройство: "cuda", "mps", "cpu" или torch.device. None — cuda, если
            доступна, иначе cpu (как раньше); "auto" — первое доступное из cuda, mps, cpu.
        max_steps : int, optional
            Обучение по шагам: ровно столько шагов оптимизатора, DataLoader перезапускается
            по исчерпании, num_epochs не используется. Расписание learning rate — на max_steps.
        eval_interval : int, optional
            Валидация каждые eval_interval шагов (нужен val_dataset). None — в конце каждой эпохи.
        eval_batches : int, optional
            Ограничить валидацию первыми eval_batches батчами (быстрая оценка).
        checkpoint_dir : str, optional
            Папка для чекпоинтов: last.pt каждые save_interval шагов и в конце обучения,
            best.pt при улучшении валидационного loss (если keep_best). Создаётся при первом сохранении.
        save_interval : int, optional
            Период сохранения last.pt в шагах (нужен checkpoint_dir). None — только в конце.
        keep_best : bool, default=True
            Сохранять best.pt при улучшении валидационного loss (нужны checkpoint_dir и val_dataset).
        log_path : str, optional
            JSON-файл с записями state.log: {"step", "epoch", "lr", "train_loss", "val_loss"};
            переписывается после каждой валидации и каждой эпохи.
        seed : int, optional
            torch.manual_seed и генератор перемешивания DataLoader: порядок батчей в эпохе k
            задаётся seed + k, поэтому после resume он тот же.

        Raises
        ------
        ValueError
            Если заданы и warmup_steps, и warmup_ratio, warmup_steps < 0 или
            warmup_ratio вне [0, 1]; max_steps, eval_interval или save_interval < 1;
            eval_interval без val_dataset; save_interval без checkpoint_dir;
            неизвестное или недоступное устройство.
        """
        if warmup_steps is not None and warmup_ratio is not None:
            raise ValueError("Задайте warmup_steps или warmup_ratio, но не оба")
        if warmup_steps is not None and warmup_steps < 0:
            raise ValueError(f"warmup_steps должен быть ≥ 0, получено {warmup_steps}")
        if warmup_ratio is not None and not 0 <= warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio должен быть от 0 до 1, получено {warmup_ratio}")
        if warmup_steps is None and warmup_ratio is None:
            warmup_steps = 100
        for name, value in (("max_steps", max_steps), ("eval_interval", eval_interval),
                            ("eval_batches", eval_batches), ("save_interval", save_interval)):
            if value is not None and value < 1:
                raise ValueError(f"{name} должен быть ≥ 1, получено {value}")
        if eval_interval is not None and val_dataset is None:
            raise ValueError("eval_interval задан, а val_dataset нет: нечего оценивать")
        if save_interval is not None and checkpoint_dir is None:
            raise ValueError("save_interval задан, а checkpoint_dir нет: некуда сохранять")

        self.seed = seed
        self._generator = None
        if seed is not None:
            torch.manual_seed(seed)
            self._generator = torch.Generator()

        self.model = model
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, generator=self._generator
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        )
        self.optimizer = get_optimizer(model, lr=lr)
        self.scheduler = None
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.keep_best = keep_best
        self.log_path = log_path

        self.state = TrainState()
        self._pending_scheduler_state = None  # из resume: применяется, когда создан scheduler

    # ------------------------------------------------------------------ состояние

    @property
    def loss_history(self):
        """Средний train loss по завершённым эпохам."""
        return self.state.loss_history

    @loss_history.setter
    def loss_history(self, value):
        self.state.loss_history = value

    @property
    def schedule_args(self):
        """Аргументы, от которых зависят число шагов и расписание; сохраняются в чекпоинт."""
        return {name: getattr(self, name) for name in SCHEDULE_ARGS}

    def num_warmup_steps(self, num_training_steps):
        """
        Число шагов warmup для обучения из num_training_steps шагов: warmup_steps
        или ceil(num_training_steps · warmup_ratio).
        """
        if self.warmup_ratio is not None:
            return math.ceil(num_training_steps * self.warmup_ratio)
        return self.warmup_steps

    def total_steps(self):
        """Число шагов оптимизатора за всё обучение: max_steps или батчей × эпох."""
        if self.max_steps is not None:
            return self.max_steps
        return len(self.train_loader) * self.num_epochs

    # ------------------------------------------------------------------ loss и forward

    def compute_lm_loss(self, logits, labels):
        """
        Loss автогрессивного языкового моделирования: cross-entropy следующего токена
        со сдвигом логитов и меток, паддинг (-100) не учитывается, батч без целей
        даёт 0, а не NaN. Реализация — `llm.training.loss.causal_lm_loss`, общая
        с `llm.evaluation`.

        Аргументы
        ---------
        logits : torch.Tensor
            Логиты модели: (batch_size, seq_len, vocab_size)
        labels : torch.Tensor
            Правильные метки: (batch_size, seq_len)
        """
        return causal_lm_loss(logits, labels)

    def _forward(self, batch):
        """
        Прямой проход по батчу: логиты модели.

        attention_mask передаётся в модель, только если она есть в батче: модели llm
        маскируют по ней паддинг (для Mixtral — и в load-balancing loss), а модели без
        этого аргумента получают один input_ids, как раньше.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            outputs = self.model(input_ids, attention_mask=attention_mask.to(self.device))
        else:
            outputs = self.model(input_ids)
        # Универсально обрабатываем выходы модели: tuple или просто tensor (logits)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _train_step(self, batch):
        """Один шаг оптимизатора: forward, loss (+ auxiliary_loss), backward, clipping, step."""
        self.optimizer.zero_grad()
        labels = batch["labels"].to(self.device)
        logits = self._forward(batch)

        # Loss автогрессивной LM-задачи и вспомогательный loss модели
        # (например, load-balancing loss роутера MoE), если он есть
        loss = self.compute_lm_loss(logits, labels)
        aux_loss = self.model.auxiliary_loss() if hasattr(self.model, "auxiliary_loss") else None
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    # ------------------------------------------------------------------ цикл обучения

    def train(self):
        """
        Запускает процесс обучения модели: по num_epochs эпохам или по max_steps шагам.

        В процессе:
        - Применяет optimizer, scheduler с warmup и decay, grad clipping (обрезка градиентов)
        - Вызывает функцию потерь для языкового моделирования
        - Показывает динамику процесса (tqdm)
        - Проводит валидацию в конце каждой эпохи или каждые eval_interval шагов
        - Сохраняет чекпоинты в checkpoint_dir, если он задан
        - После resume продолжает с сохранённого шага: уже пройденные батчи эпохи пропускаются

        Параметры задаются на этапе инициализации Trainer.
        """
        total_steps = self.total_steps()
        warmup_steps = self.num_warmup_steps(total_steps)
        if warmup_steps > 0 and warmup_steps >= total_steps:
            # Всё обучение внутри warmup: learning rate не дойдёт до заданного
            warnings.warn(
                f"warmup_steps = {warmup_steps} не меньше числа шагов обучения {total_steps}: "
                f"learning rate не поднимется выше {(total_steps - 1) / max(1, warmup_steps):.2f} "
                f"от заданного. Уменьшите warmup_steps или задайте warmup_ratio."
            )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        if self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None
            # Конструктор LambdaLR выставил lr начала warmup; вернуть lr сохранённого шага
            for group, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr()):
                group["lr"] = lr

        state = self.state
        window_loss, window_steps = 0.0, 0  # для train_loss в записях лога

        while state.step < total_steps:
            self.model.train()
            if self._generator is not None:
                # Порядок батчей эпохи k зависит только от seed + k: после resume он тот же
                self._generator.manual_seed(self.seed + state.epoch)
            epoch_loss, epoch_steps = 0.0, 0
            save_due = False
            desc = (
                f"Epoch {state.epoch + 1}/{self.num_epochs}" if self.max_steps is None
                else f"Epoch {state.epoch + 1}"
            )
            progress_bar = tqdm(self.train_loader, desc=desc)
            epoch_finished = True

            for batch_idx, batch in enumerate(progress_bar):
                if batch_idx < state.step_in_epoch:
                    continue  # уже пройдено до resume
                loss = self._train_step(batch)
                state.step += 1
                state.step_in_epoch += 1
                epoch_loss += loss
                epoch_steps += 1
                window_loss += loss
                window_steps += 1
                progress_bar.set_postfix(loss=loss)

                finished = state.step >= total_steps
                if self.eval_interval is not None and (
                    state.step % self.eval_interval == 0 or finished
                ):
                    self._log_and_evaluate(window_loss / window_steps)
                    window_loss, window_steps = 0.0, 0
                    if not finished:
                        self.model.train()
                if self.save_interval is not None and state.step % self.save_interval == 0:
                    if batch_idx + 1 == len(self.train_loader):
                        # Последний батч эпохи: сохранить после закрытия эпохи, чтобы в
                        # чекпоинт попали её loss_history, лог и номер следующей эпохи
                        save_due = True
                    else:
                        self.save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"))
                if finished:
                    epoch_finished = batch_idx + 1 == len(self.train_loader)
                    break

            if epoch_steps:
                avg_loss = epoch_loss / epoch_steps
                print(f"Epoch {state.epoch + 1} finished — avg loss: {avg_loss:.4f}")
                if epoch_finished or self.max_steps is not None:
                    state.loss_history.append(avg_loss)
            if epoch_finished:
                if self.eval_interval is None:
                    self._log_and_evaluate(window_loss / window_steps if window_steps else None)
                    window_loss, window_steps = 0.0, 0
                state.epoch += 1
                state.step_in_epoch = 0
            if save_due and state.step < total_steps:  # в конце обучения last.pt пишется ниже
                self.save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"))

        if self.checkpoint_dir is not None:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"))

    def _log_and_evaluate(self, train_loss):
        """Запись в лог с текущим lr и train_loss; с val_loader — ещё и валидация и best.pt."""
        record = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "lr": self.optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
        }
        if self.val_loader:
            val_loss = self.evaluate()
            record["val_loss"] = val_loss
            if self.state.best_val_loss is None or val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss
                if self.keep_best and self.checkpoint_dir is not None:
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best.pt"))
        self.state.log.append(record)
        self._write_log()

    def _write_log(self):
        if self.log_path is None:
            return
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.state.log, f, ensure_ascii=False, indent=2)

    def evaluate(self):
        """
        Оценивает модель на валидационном датасете (если задан).

        В режиме eval() модели отключается dropout и все стохастические элементы.
        Возвращает среднее значение функции потерь (loss) по всему validation set
        (или по первым eval_batches батчам). Модель остаётся в режиме eval.
        """
        self.model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if self.eval_batches is not None and count >= self.eval_batches:
                    break
                labels = batch["labels"].to(self.device)
                logits = self._forward(batch)
                loss = self.compute_lm_loss(logits, labels)
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    # ------------------------------------------------------------------ чекпоинты

    def save_checkpoint(self, path):
        """
        Сохраняет модель и состояние обучения в path (см. llm/training/checkpoint.py).
        Файл читается и как модель (`BaseModel.load`), и для продолжения (`resume`).
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        save_checkpoint(path, self.model, self.optimizer, self.scheduler, self.state, self.schedule_args)

    def resume(self, path):
        """
        Восстанавливает из чекпоинта веса, оптимизатор, планировщик, шаг, историю и
        генератор случайных чисел; следующий `train()` продолжит с сохранённого шага.

        Trainer должен быть создан с той же моделью и теми же аргументами расписания
        (lr, batch_size, num_epochs, max_steps, warmup_steps, warmup_ratio) — иначе
        число шагов и learning rate не совпадут с сохранённым планировщиком.

        Returns:
            self — чтобы писать `Trainer(...).resume(path).train()`.

        Raises:
            ValueError: другой класс модели, другие аргументы расписания или не чекпоинт обучения.
        """
        checkpoint = load_checkpoint(path)
        model_class = type(self.model).__name__
        if checkpoint["model_class"] != model_class:
            raise ValueError(
                f"{path}: чекпоинт модели {checkpoint['model_class']}, а обучается {model_class}"
            )
        saved_args = checkpoint["trainer"]["args"]
        current_args = self.schedule_args
        diff = {k: (saved_args.get(k), current_args[k]) for k in SCHEDULE_ARGS
                if saved_args.get(k) != current_args[k]}
        if diff:
            details = ", ".join(f"{k}: в чекпоинте {a!r}, сейчас {b!r}" for k, (a, b) in diff.items())
            raise ValueError(f"{path}: аргументы расписания не совпадают — {details}")

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["trainer"]["optimizer"])
        self._pending_scheduler_state = checkpoint["trainer"]["scheduler"]
        self.state = TrainState(**checkpoint["trainer"]["state"])
        set_rng_state(checkpoint["trainer"]["rng"])
        return self
