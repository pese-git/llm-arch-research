"""
Модуль токенизаторов для библиотеки llm.

Предоставляет различные реализации токенизаторов:
- BPE (Byte Pair Encoding) токенизатор
- Базовый интерфейс для создания собственных токенизаторов

Примеры использования:
    >>> from llm.tokenizers import BPETokenizer, SimpleBPETokenizer
    >>> tokenizer = BPETokenizer()
    >>> tokenizer.train(["текст для обучения", "еще текст"])
    >>> tokens = tokenizer.encode("привет мир")
    >>> text = tokenizer.decode(tokens)
"""

from .base_tokenizer import BaseTokenizer
from .bpe_tokenizer import BPETokenizer, SimpleBPETokenizer

__all__ = ["BaseTokenizer", "BPETokenizer", "SimpleBPETokenizer"]
