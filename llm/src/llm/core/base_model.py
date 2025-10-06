# llm/core/base_model.py
"""
Базовый абстрактный класс для всех языковых моделей (LLM).

Реализует общий интерфейс для прямого прохода и генерации текста.
Все конкретные модели должны наследоваться от этого класса.
"""

import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class BaseModel(nn.Module, ABC):
    """
    Абстрактный базовый класс для больших языковых моделей.
    
    Args:
        config (dict): Конфигурация модели с параметрами архитектуры
    
    Attributes:
        config (dict): Конфигурационные параметры модели
    """

    def __init__(self, config: dict):
        """
        Инициализация базовой модели.
        
        Args:
            config: Словарь с параметрами конфигурации модели
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            input_ids: Тензор индексов токенов формы [batch_size, seq_len]
            attention_mask: Опциональная маска внимания формы [batch_size, seq_len]
            
        Returns:
            Тензор логитов формы [batch_size, seq_len, vocab_size]
        """
        pass

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        """
        Генерация текста с использованием greedy decoding или sampling.
        
        Args:
            input_ids: Начальные токены для генерации формы [batch_size, start_len]
            max_length: Максимальная длина генерируемой последовательности
            
        Returns:
            Тензор сгенерированных токенов формы [batch_size, generated_len]
        """
        pass
