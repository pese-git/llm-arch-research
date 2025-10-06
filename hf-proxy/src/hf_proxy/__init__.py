"""
HF-Proxy: Адаптер для интеграции моделей llm с HuggingFace Transformers.

Этот пакет предоставляет инструменты для:
- Конвертации кастомных LLM моделей в формат HuggingFace
- Использования моделей через стандартные интерфейсы Transformers
- Загрузки моделей в HuggingFace Hub
- Создания pipelines для генерации текста

Основные классы:
- HFAdapter: Главный адаптер для преобразования моделей
- HFGPTAdapter: Адаптер для GPT моделей
- HFUtils: Утилиты для работы с адаптером
- HFTokenizerAdapter: Адаптер для кастомных токенизаторов
"""

from .hf_adapter import HFAdapter, HFGPTAdapter
from .hf_config import HFAdapterConfig, HFPretrainedConfig
from .hf_utils import HFUtils, TokenizerWrapper, create_hf_pipeline
from .hf_tokenizer import HFTokenizerAdapter, create_hf_tokenizer, convert_to_hf_format

__version__ = "0.2.0"
__author__ = "Sergey Penkovsky"
__email__ = "sergey.penkovsky@gmail.com"

__all__ = [
    # Основные классы адаптера
    "HFAdapter",
    "HFGPTAdapter",
    # Конфигурации
    "HFAdapterConfig",
    "HFPretrainedConfig",
    # Адаптеры токенизаторов
    "HFTokenizerAdapter",
    "create_hf_tokenizer",
    "convert_to_hf_format",
    # Утилиты
    "HFUtils",
    "TokenizerWrapper",
    "create_hf_pipeline",
]
