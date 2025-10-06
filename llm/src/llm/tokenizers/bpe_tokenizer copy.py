"""
BPE (Byte Pair Encoding) токенизатор.

Реализация алгоритма BPE для токенизации текста.
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE токенизатор для обработки текста.

    Реализует алгоритм Byte Pair Encoding для создания субсловных токенов.

    Примеры использования:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train(["пример текста для обучения"], vocab_size=1000)
        >>> tokens = tokenizer.encode("новый текст")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(self):
        super().__init__()
        self.merges: Dict[Tuple[str, str], int] = {}
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern, re.UNICODE)

    def train(self, texts: List[str], vocab_size: int = 1000, **kwargs):
        """
        Обучение BPE токенизатора на текстах.

        Args:
            texts: Список текстов для обучения
            vocab_size: Желаемый размер словаря
            **kwargs: Дополнительные параметры
                - min_frequency: Минимальная частота для мерджа
                - special_tokens: Список специальных токенов
        """
        # Инициализация базового словаря
        self._initialize_vocab()

        # Добавляем специальные токены если указаны
        special_tokens = kwargs.get(
            "special_tokens",
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token],
        )
        self.add_special_tokens(special_tokens)

        # Предобработка текстов
        words = self._preprocess_texts(texts)

        # Получаем начальные токены
        vocab = self._get_initial_vocab(words)

        # Выполняем BPE мерджи
        self._perform_merges(vocab, vocab_size, kwargs.get("min_frequency", 2))

        # Строим финальный словарь
        self._build_final_vocab()

    def _initialize_vocab(self):
        """Инициализирует базовый словарь."""
        self.vocab.clear()
        self.inverse_vocab.clear()
        self.merges.clear()
        self.vocab_size = 0

    def _preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Предобработка текстов для обучения.

        Args:
            texts: Список текстов

        Returns:
            List[List[str]]: Предобработанные слова
        """
        words = []
        for text in texts:
            # Базовая нормализация
            text = text.lower().strip()
            # Токенизация на слова
            tokens = self.compiled_pattern.findall(text)
            words.append(tokens)
        return words

    def _get_initial_vocab(self, words: List[List[str]]) -> Dict[str, int]:
        """
        Создает начальный словарь из символов.

        Args:
            words: Список токенизированных текстов

        Returns:
            Dict[str, int]: Начальный словарь частот
        """
        vocab = Counter()
        for word_list in words:
            for word in word_list:
                # Разбиваем слово на символы и добавляем специальный символ конца слова
                chars = list(word) + ["</w>"]
                vocab.update(["".join(chars[i : i + 1]) for i in range(len(chars))])
        return vocab

    def _perform_merges(
        self, vocab: Dict[str, int], target_vocab_size: int, min_frequency: int
    ):
        """
        Выполняет BPE мерджи до достижения целевого размера словаря.

        Args:
            vocab: Начальный словарь
            target_vocab_size: Целевой размер словаря
            min_frequency: Минимальная частота для мерджа
        """
        current_vocab_size = len(vocab) + len(self.vocab)

        while current_vocab_size < target_vocab_size:
            # Находим наиболее частую пару
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break

            # Выполняем мердж
            vocab = self._merge_vocab(vocab, best_pair)
            self.merges[best_pair] = len(self.merges)
            current_vocab_size += 1

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Собирает статистику по парам символов.

        Args:
            vocab: Словарь токенов

        Returns:
            Dict[Tuple[str, str], int]: Частоты пар
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge_vocab(
        self, vocab: Dict[str, int], pair: Tuple[str, str]
    ) -> Dict[str, int]:
        """
        Объединяет пару символов в словаре.

        Args:
            vocab: Исходный словарь
            pair: Пара для объединения

        Returns:
            Dict[str, int]: Обновленный словарь
        """
        new_vocab = {}
        bigram = re.compile(
            r"(?<!\\S)" + re.escape(pair[0]) + r" " + re.escape(pair[1]) + r"(?!\\S)"
        )
        replacement = pair[0] + pair[1]

        for word in vocab:
            new_word = bigram.sub(replacement, word)
            new_vocab[new_word] = vocab[word]

        return new_vocab

    def _build_final_vocab(self):
        """Строит финальный словарь токенизатора."""
        # Собираем все уникальные токены из мерджей
        all_tokens = set()

        # Добавляем специальные токены
        all_tokens.update(
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        )

        # Добавляем токены из мерджей
        for pair in self.merges:
            all_tokens.update(pair)

        # Создаем словарь
        for i, token in enumerate(sorted(all_tokens)):
            self.vocab[token] = i

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Обновляем ID специальных токенов
        self.pad_token_id = self.vocab.get(self.pad_token)
        self.unk_token_id = self.vocab.get(self.unk_token)
        self.bos_token_id = self.vocab.get(self.bos_token)
        self.eos_token_id = self.vocab.get(self.eos_token)

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Кодирует текст в последовательность токенов.

        Args:
            text: Входной текст
            **kwargs: Дополнительные параметры
                - add_special_tokens: Добавлять специальные токены

        Returns:
            List[int]: Список идентификаторов токенов
        """
        add_special_tokens = kwargs.get("add_special_tokens", False)

        # Токенизация текста
        tokens = self.compiled_pattern.findall(text)

        # Применяем BPE к каждому токену
        bpe_tokens = []
        for token in tokens:
            # Преобразуем токен в BPE представление
            bpe_token = self._apply_bpe(token)
            bpe_tokens.extend(bpe_token)

        # Конвертируем в ID
        token_ids = []
        for token in bpe_tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            if token_id is not None:
                token_ids.append(token_id)

        # Добавляем специальные токены если нужно
        if add_special_tokens:
            if self.bos_token_id is not None:
                token_ids.insert(0, self.bos_token_id)
            if self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)

        return token_ids

    def _apply_bpe(self, token: str) -> List[str]:
        """
        Применяет BPE к одному токену.

        Args:
            token: Входной токен

        Returns:
            List[str]: Список BPE токенов
        """
        # Простая реализация - в реальной реализации нужно применять обученные мерджи
        word = token + "</w>"
        tokens = [word[i : i + 1] for i in range(len(word))]

        # Применяем мерджи (упрощенная версия)
        # В полной реализации нужно применять все обученные мерджи
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = tokens[i] + tokens[i + 1]
                    del tokens[i + 1]
                else:
                    i += 1

        return tokens

    def decode(self, tokens: List[int], **kwargs) -> str:
        """
        Декодирует последовательность токенов в текст.

        Args:
            tokens: Список идентификаторов токенов
            **kwargs: Дополнительные параметры
                - skip_special_tokens: Пропускать специальные токены

        Returns:
            str: Декодированный текст
        """
        skip_special_tokens = kwargs.get("skip_special_tokens", True)

        # Конвертируем ID в токены
        token_strings = []
        for token_id in tokens:
            token = self.inverse_vocab.get(token_id, self.unk_token)

            # Пропускаем специальные токены если нужно
            if skip_special_tokens and token in [
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token,
            ]:
                continue

            token_strings.append(token)

        # Объединяем токены в текст
        text = "".join(token_strings)

        # Убираем маркер конца слова
        text = text.replace("</w>", " ")

        return text.strip()

    def save(self, filepath: str):
        """
        Сохраняет BPE токенизатор в файл.

        Args:
            filepath: Путь для сохранения
        """
        import json

        config = {
            "vocab": self.vocab,
            "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pattern": self.pattern,
            "tokenizer_type": self.__class__.__name__,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """
        Загружает BPE токенизатор из файла.

        Args:
            filepath: Путь к файлу

        Returns:
            BPETokenizer: Загруженный токенизатор
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = config["vocab"]
        tokenizer.vocab_size = config["vocab_size"]
        tokenizer.pad_token = config["pad_token"]
        tokenizer.unk_token = config["unk_token"]
        tokenizer.bos_token = config["bos_token"]
        tokenizer.eos_token = config["eos_token"]
        tokenizer.pattern = config.get("pattern", tokenizer.pattern)
        tokenizer.compiled_pattern = re.compile(tokenizer.pattern, re.UNICODE)

        # Восстанавливаем мерджи
        merges = config.get("merges", {})
        tokenizer.merges = {}
        for k, v in merges.items():
            parts = k.split()
            if len(parts) == 2:
                tokenizer.merges[(parts[0], parts[1])] = v

        # Создаем обратный словарь
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

        # Обновляем ID специальных токенов
        tokenizer.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token)
        tokenizer.unk_token_id = tokenizer.vocab.get(tokenizer.unk_token)
        tokenizer.bos_token_id = tokenizer.vocab.get(tokenizer.bos_token)
        tokenizer.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token)

        return tokenizer


# Упрощенная версия для быстрого старта
class SimpleBPETokenizer(BPETokenizer):
    """
    Упрощенная версия BPE токенизатора для демонстрации.
    """

    def train(self, texts: List[str], vocab_size: int = 1000, **kwargs):
        """Упрощенное обучение для демонстрации."""
        # Инициализация базового словаря
        self._initialize_vocab()

        # Добавляем базовые токены
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
        self.add_special_tokens(special_tokens)

        # Простая реализация - собираем все символы
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        # Добавляем символы в словарь
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Обновляем ID специальных токенов
        self.pad_token_id = self.vocab.get(self.pad_token)
        self.unk_token_id = self.vocab.get(self.unk_token)
        self.bos_token_id = self.vocab.get(self.bos_token)
        self.eos_token_id = self.vocab.get(self.eos_token)

    def encode(self, text: str, **kwargs) -> List[int]:
        """Упрощенное кодирование - разбиваем на символы."""
        add_special_tokens = kwargs.get("add_special_tokens", False)

        token_ids = []
        for char in text:
            token_id = self.vocab.get(char, self.unk_token_id)
            if token_id is not None:
                token_ids.append(token_id)

        if add_special_tokens:
            if self.bos_token_id is not None:
                token_ids.insert(0, self.bos_token_id)
            if self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, tokens: List[int], **kwargs) -> str:
        """Упрощенное декодирование."""
        skip_special_tokens = kwargs.get("skip_special_tokens", True)

        chars = []
        for token_id in tokens:
            char = self.inverse_vocab.get(token_id, self.unk_token)
            if skip_special_tokens and char in [
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token,
            ]:
                continue
            chars.append(char)

        return "".join(chars)
