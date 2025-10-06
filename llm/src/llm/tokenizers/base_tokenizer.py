"""
Базовый класс для токенизаторов.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class BaseTokenizer(ABC):
    """
    Абстрактный базовый класс для всех токенизаторов.

    Определяет общий интерфейс для токенизации текста.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.vocab_size: int = 0

        # Специальные токены
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.pad_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None

    @abstractmethod
    def train(self, texts: List[str], vocab_size: int = 1000, **kwargs):
        """
        Обучение токенизатора на текстах.

        Args:
            texts: Список текстов для обучения
            vocab_size: Желаемый размер словаря
            **kwargs: Дополнительные параметры обучения
        """
        pass

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Кодирование текста в последовательность токенов.

        Args:
            text: Входной текст
            **kwargs: Дополнительные параметры кодирования

        Returns:
            List[int]: Список идентификаторов токенов
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int], **kwargs) -> str:
        """
        Декодирование последовательности токенов в текст.

        Args:
            tokens: Список идентификаторов токенов
            **kwargs: Дополнительные параметры декодирования

        Returns:
            str: Декодированный текст
        """
        pass

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Токенизация текста в список строковых токенов.

        Args:
            text: Входной текст
            **kwargs: Дополнительные параметры

        Returns:
            List[str]: Список токенов
        """
        token_ids = self.encode(text, **kwargs)
        return [
            self.inverse_vocab.get(token_id, self.unk_token) for token_id in token_ids
        ]

    def get_vocab(self) -> Dict[str, int]:
        """Возвращает словарь токенизатора."""
        return self.vocab.copy()

    def get_vocab_size(self) -> int:
        """Возвращает размер словаря."""
        return self.vocab_size

    def add_special_tokens(self, special_tokens: List[str]):
        """
        Добавляет специальные токены в словарь.

        Args:
            special_tokens: Список специальных токенов
        """
        for token in special_tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.inverse_vocab[token_id] = token
                self.vocab_size += 1

        # Обновляем ID специальных токенов
        self.pad_token_id = self.vocab.get(self.pad_token)
        self.unk_token_id = self.vocab.get(self.unk_token)
        self.bos_token_id = self.vocab.get(self.bos_token)
        self.eos_token_id = self.vocab.get(self.eos_token)

    def save(self, filepath: str):
        """
        Сохраняет токенизатор в файл.

        Args:
            filepath: Путь для сохранения
        """
        config = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "tokenizer_type": self.__class__.__name__,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """
        Загружает токенизатор из файла.

        Args:
            filepath: Путь к файлу

        Returns:
            BaseTokenizer: Загруженный токенизатор
        """
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Создаем экземпляр токенизатора
        tokenizer = cls()
        tokenizer.vocab = config["vocab"]
        tokenizer.vocab_size = config["vocab_size"]
        tokenizer.pad_token = config["pad_token"]
        tokenizer.unk_token = config["unk_token"]
        tokenizer.bos_token = config["bos_token"]
        tokenizer.eos_token = config["eos_token"]

        # Создаем обратный словарь
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

        # Обновляем ID специальных токенов
        tokenizer.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token)
        tokenizer.unk_token_id = tokenizer.vocab.get(tokenizer.unk_token)
        tokenizer.bos_token_id = tokenizer.vocab.get(tokenizer.bos_token)
        tokenizer.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token)

        return tokenizer

    def __len__(self) -> int:
        """Возвращает размер словаря."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
