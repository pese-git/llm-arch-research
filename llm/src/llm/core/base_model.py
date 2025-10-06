"""
Базовый абстрактный класс для всех больших языковых моделей (LLM).

Научная суть:
Модели типа LLM строятся по модульному принципу — конкретные GPT, LLaMA и др. должны наследоваться от этого класса и реализовывать базовый набор интерфейсов для совместимости с training loop, генерацией, инференсом и т.д.

Пользовательский уровень:
Базовый интерфейс минимизирует дублирование кода и позволяет быстро добавлять новые архитектуры.

Использование:
    class MyModel(BaseModel):
        ...
    model = MyModel(config)
    logits = model.forward(input_ids)
    tokens = model.generate(input_ids)
"""
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseModel(nn.Module, ABC):
    """
    Абстрактный класс — стандарт для всех архитектур LLM.

    Научная идея:
    Реализация унифицированного входа/выхода для поддержки построения и обучения любых современных языковых моделей.

    Args:
        config (dict): Параметры архитектуры (размерность эмбеддингов, число слоев, heads и т.д.)

    Attributes:
        config (dict): Конфиг модели
    """

    def __init__(self, config: dict):
        """
        Инициализация модели.

        Args:
            config (dict): Настройки архитектуры модели (размеры слоев, типы блоков и т.д.)
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход — получение логитов для входных токенов.

        Args:
            input_ids (Tensor[int]): Индексы токенов [batch, seq_len]
            attention_mask (Optional[Tensor[bool]]): Маска разрешенных позиций (если требуется) [batch, seq_len]
        Returns:
            logits (Tensor[float]): Логиты словаря [batch, seq_len, vocab_size]
        """
        pass

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        """
        Генерация текста (авторегрессивно, greedy или sampling).

        Args:
            input_ids (Tensor[int]): Начальные токены [batch, start_len]
            max_length (int): Максимальная длина последовательности
        Returns:
            output_tokens (Tensor[int]): Сгенерированная последовательность [batch, generated_len]
        Пример:
            >>> logits = model.forward(input_ids)
            >>> generated = model.generate(input_ids, max_length=128)
        """
        pass
