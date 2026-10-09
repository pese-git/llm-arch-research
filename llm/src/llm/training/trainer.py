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
"""
import math
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from llm.training.loss import causal_lm_loss
from llm.training.optimizer import get_optimizer
from llm.training.scheduler import get_linear_schedule_with_warmup


class Trainer:
    """
    Универсальный и расширяемый класс для обучения больших языковых моделей (Large Language Models, LLM).

    Поддерживаются архитектуры семейства GPT, LLaMA, Mistral и другие автогрессивные модели.
    Объединяет:
      - Тренировку по задаче языкового моделирования (Causal LM)
      - Cross-entropy loss с автоматическим сдвигом логитов/меток
      - Поддержку Grad Clipping, Scheduler, Validation
      - Унифицированный даталоадер, автоматический выбор устройства (CPU/GPU)

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
        Устройство (CPU или CUDA), куда помещается модель
    num_epochs : int
        Количество эпох обучения
    warmup_steps : int или None
        Число шагов warmup для scheduler (None, если задан warmup_ratio)
    warmup_ratio : float или None
        Доля warmup от общего числа шагов (None, если задан warmup_steps)
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
            Количество эпох обучения.
        warmup_steps : int, optional
            Количество шагов разогрева (warmup) learning rate. Если не задан ни он,
            ни warmup_ratio — 100.
        warmup_ratio : float, optional
            Warmup как доля от общего числа шагов, от 0 до 1: ceil(N_steps · warmup_ratio),
            как warmup_ratio в HuggingFace TrainingArguments. Удобнее warmup_steps, когда
            число шагов зависит от размера датасета. Нельзя вместе с warmup_steps.

        Raises
        ------
        ValueError
            Если заданы и warmup_steps, и warmup_ratio, warmup_steps < 0 или
            warmup_ratio вне [0, 1].
        """
        if warmup_steps is not None and warmup_ratio is not None:
            raise ValueError("Задайте warmup_steps или warmup_ratio, но не оба")
        if warmup_steps is not None and warmup_steps < 0:
            raise ValueError(f"warmup_steps должен быть ≥ 0, получено {warmup_steps}")
        if warmup_ratio is not None and not 0 <= warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio должен быть от 0 до 1, получено {warmup_ratio}")
        if warmup_steps is None and warmup_ratio is None:
            warmup_steps = 100
        self.model = model
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        )
        self.optimizer = get_optimizer(model, lr=lr)
        self.scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio

    def num_warmup_steps(self, num_training_steps):
        """
        Число шагов warmup для обучения из num_training_steps шагов: warmup_steps
        или ceil(num_training_steps · warmup_ratio).
        """
        if self.warmup_ratio is not None:
            return math.ceil(num_training_steps * self.warmup_ratio)
        return self.warmup_steps

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

    def train(self):
        """
        Запускает процесс обучения модели по заданному числу эпох.

        В процессе:
        - Применяет optimizer, scheduler с warmup и decay, grad clipping (обрезка градиентов)
        - Вызывает функцию потерь для языкового моделирования
        - Показывает динамику процесса (tqdm)
        - После каждой эпохи возможно проведение валидации

        Параметры задаются на этапе инициализации Trainer.
        """
        total_steps = len(self.train_loader) * self.num_epochs
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
        self.loss_history = []  # добавлено: лог средних потерь

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            for batch in progress_bar:
                self.optimizer.zero_grad()

                labels = batch["labels"].to(self.device)
                logits = self._forward(batch)

                # Вычисляем loss автогрессивной LM-задачи и вспомогательный loss модели
                # (например, load-balancing loss роутера MoE), если он есть
                loss = self.compute_lm_loss(logits, labels)
                aux_loss = self.model.auxiliary_loss() if hasattr(self.model, "auxiliary_loss") else None
                if aux_loss is not None:
                    loss = loss + aux_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            self.loss_history.append(avg_loss)  # добавлено: запоминаем loss
            print(f"Epoch {epoch+1} finished — avg loss: {avg_loss:.4f}")

            if self.val_loader:
                self.evaluate()

    def evaluate(self):
        """
        Оценивает модель на валидационном датасете (если задан).

        В режиме eval() модели отключается dropout и все стохастические элементы.
        Возвращает среднее значение функции потерь (loss) по всему validation set.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                labels = batch["labels"].to(self.device)
                logits = self._forward(batch)
                loss = self.compute_lm_loss(logits, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss