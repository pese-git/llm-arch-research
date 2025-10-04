# llm/core/base_model.py
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Базовый класс для всех LLM."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, input_ids, attention_mask=None):
        """Прямой проход модели."""
        pass

    @abstractmethod
    def generate(self, input_ids, max_length=50):
        """Генерация текста (greedy или sampling)."""
        pass
