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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    warmup_steps : int
        Число шагов warmup для scheduler
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        lr=3e-4,
        batch_size=8,
        num_epochs=3,
        warmup_steps=100,
    ):
        """
        Инициализация обучающего класса Trainer.

        Аргументы
        ---------
        model : torch.nn.Module
            Модель для обучения (например, GPT, LLaMA, Mistral).
        train_dataset : torch.utils.data.Dataset
            Обучающий датасет с полями input_ids и labels.
        val_dataset : torch.utils.data.Dataset, optional
            Валидационный датасет для контроля качества обучения.
        lr : float, default=3e-4
            Начальный шаг обучения.
        batch_size : int, default=8
            Размер обучающего мини-батча.
        num_epochs : int, default=3
            Количество эпох обучения.
        warmup_steps : int, default=100
            Количество шагов разогрева (warmup) learning rate.
        """
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

    def compute_lm_loss(self, logits, labels):
        """
        Вычисляет функцию потерь (loss) для задачи автогрессивного языкового моделирования.

        Производит сдвиг логитов и меток: предсказания делаются для следующего токена.
        Используется кросс-энтропия (CrossEntropyLoss), что соответствует максимизации логарифма правдоподобия:
            L = -log P(w_{t+1} | w_1,...,w_t)

        Аргументы
        ---------
        logits : torch.Tensor
            Логиты модели: (batch_size, seq_len, vocab_size)
        labels : torch.Tensor
            Правильные метки: (batch_size, seq_len)
        Возвращаемое значение
        ---------------------
        loss : torch.Tensor
            Средний loss по batch.
        """
        # Сдвигаем логиты и метки для языкового моделирования (автогрессия)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # CrossEntropyLoss (игнорируем паддинги: ignore_index=-100)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # Padding токены не участвуют в loss
        )
        return loss

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
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, self.warmup_steps, total_steps
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

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Универсально обрабатываем выходы модели: tuple или просто tensor (logits)
                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Вычисляем loss автогрессивной LM-задачи
                loss = self.compute_lm_loss(logits, labels)
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
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = self.compute_lm_loss(logits, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation loss: {avg_loss:.4f}")