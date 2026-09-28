"""
BPE (Byte Pair Encoding) токенизатор.

Реализация алгоритма BPE для токенизации текста.
"""

import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
from .base_tokenizer import BaseTokenizer

# Разбиение текста на слова перед BPE (как в GPT-2): пробел прикрепляется
# к началу следующего слова, пунктуация идет отдельными кусками, остальные
# пробельные символы — своими. Конкатенация кусков дает исходный текст.
_PRETOKENIZE_PATTERN = re.compile(r" ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> List[str]:
    """
    Разбивает текст на слова, внутри которых работает BPE.

    Пример:
        >>> pretokenize("Привет, мир!")
        ['Привет', ',', ' мир', '!']
    """
    return _PRETOKENIZE_PATTERN.findall(text)


class BPETokenizer(BaseTokenizer):
    """
    BpeTokenizer — реализация токенизатора на алгоритме byte pair encoding (BPE).

    Назначение:
    -----------
    - Преобразует открытый текст (строки, bytes) в последовательность числовых токенов для подачи в LLM и обратно.
    - Разбивает текст на сабслова (байтовые пары), эффективно кодируя редкие слова длинными последовательностями, а частые — единичными токенами.
    - Является стандартом де-факто в современных языковых моделях (GPT, LLaMA, BLOOM, Mistral, HuggingFace).

    Как работает BPE:
    -----------------
    1. Строится словарь из наиболее популярных пар символов/субстрок.
    2. Текст замещается наиболее длинными subword-подстроками из vocabulary (жадно).
    3. Итог: многомиллионное лексическое пространство сокращается до компактного набора subword pieces.

    Особенности алгоритма:
    ----------------------
    - Отлично работает на всех языках, включая rare/compound/inflectable.
    - Гибко масштабируется под размер итогового словаря/token space.
    - Обычно хранит mapping (str/bytes → int и int → str/bytes) в JSON или словарном файле.
    - Может использовать кастомные сепараторы, handle unknown.

    Аргументы конструктора:
    -----------------------
    vocab_path: str
        Путь к файлу BPE vocabulary (JSON, txt, в зависимости от реализации).
    merges_path: str, optional
        Путь к списку merge-правил (если используется блочное файловое раздельное хранение).
    unk_token: str, optional
        Токен для неизвестных последовательностей (по дефолту '[UNK]' или '<unk>').
    pad_token, bos_token, eos_token: str, optional
        Special tokens, если нужны для вашей архитектуры.
    lowercase: bool, optional
        Приводить ли текст к нижнему регистру перед токенизацией.

    Пример:
    -------
        >>> tokenizer = BpeTokenizer(vocab_path=\"bpe_vocab.json\")
        >>> tokens = tokenizer.encode(\"Hello, world!\")
        >>> print(tokens)  # [15496, 11, ...]
        >>> text = tokenizer.decode(tokens)
        >>> print(text)  # 'Hello, world!'

    References:
    -----------
    - Sennrich et al, \"Neural Machine Translation of Rare Words with Subword Units\", 2015: https://arxiv.org/abs/1508.07909
    - GPT-2 tokenization: https://github.com/openai/gpt-2
    - HuggingFace tokenizers overview: https://huggingface.co/docs/tokenizers/index
    - Visually: https://guillaume-be.github.io/2021-05-21/byte-pair-encoding/
    """

    def __init__(self):
        super().__init__()
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab_list: List[str] = []

    def train(self, texts: List[str], vocab_size: int = 1000, **kwargs):
        """
        Обучение BPE токенизатора на текстах.

        Перед обучением каждый текст разбивается на слова (см. pretokenize):
        пары сливаются только внутри слова, поэтому токены не пересекают
        границы слов и текстов — как в GPT-2 и у Sennrich et al.

        Args:
            texts: Список текстов для обучения
            vocab_size: Желаемый размер словаря (без специальных токенов).
                Обучение останавливается раньше, если каждое слово уже
                стало одним токеном.
            **kwargs: Дополнительные параметры
                - special_tokens: Список специальных токенов
        """
        # 1. Разбиваем тексты на слова и считаем частоты слов
        word_freq = Counter()
        for text in texts:
            word_freq.update(pretokenize(text))

        # 2. Начальный словарь — уникальные символы
        tokens = sorted({char for word in word_freq for char in word})
        known_tokens = set(tokens)

        # Каждое уникальное слово — последовательность токенов и его частота
        words = [(list(word), freq) for word, freq in word_freq.items()]
        self.merges = {}

        # 3. Объединяем токены до достижения нужного размера словаря
        while len(tokens) < vocab_size:
            # Считаем частоты пар внутри слов с учетом частоты слова
            pair_freq: Dict[Tuple[str, str], int] = {}
            for symbols, freq in words:
                for pair in zip(symbols, symbols[1:]):
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq

            if not pair_freq:
                break  # каждое слово уже стало одним токеном

            # Самая частая пара; при равенстве max берет первую встреченную
            most_frequent_pair = max(pair_freq, key=pair_freq.get)
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            self.merges[most_frequent_pair] = len(self.merges)

            # Одна и та же строка может получиться разными слияниями
            # ("а"+"бв" и "аб"+"в"): в словарь добавляем ее один раз
            if new_token not in known_tokens:
                tokens.append(new_token)
                known_tokens.add(new_token)

            for symbols, _ in words:
                symbols[:] = self._merge_pair(symbols, most_frequent_pair, new_token)

        # 4. Создаем словари
        self.vocab_list = tokens.copy()
        self.vocab = {token: i for i, token in enumerate(tokens)}
        self.inverse_vocab = {i: token for i, token in enumerate(tokens)}
        self.vocab_size = len(self.vocab)

        # Добавляем специальные токены если указаны
        special_tokens = kwargs.get(
            "special_tokens",
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token],
        )
        self.add_special_tokens(special_tokens)

    @staticmethod
    def _merge_pair(
        symbols: List[str], pair: Tuple[str, str], new_token: str
    ) -> List[str]:
        """Заменяет все вхождения пары в последовательности на новый токен."""
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(new_token)
                i += 2  # пропускаем два символа — заменённую пару
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Токенизирует входной текст в список числовых токенов (индексов).

        Args:
        -----
        text: str
            Входная строка/текст для токенизации.

        Returns:
        --------
        List[int] — последовательность индексов из vocabulary.

        Пример:
        -------
            >>> ids = tokenizer.encode(\"The quick brown fox\")
            >>> print(ids)
        """
        add_special_tokens = kwargs.get("add_special_tokens", False)

        # 1. Разбиваем текст на слова так же, как при обучении
        tokens = []
        for word in pretokenize(text):
            # 2. Внутри слова жадно берем самые длинные токены
            i = 0
            while i < len(word):
                # 2.1 Найти все токены в словаре, начинающиеся с word[i]
                start_char = word[i]
                result = [
                    token for token in self.vocab_list if token.startswith(start_char)
                ]
                # 2.2 Выбрать самый длинный подходящий токен
                find_token = self._find_max_matching_token(word[i:], result)
                if find_token is None:
                    # Обработка неизвестного символа
                    tokens.append(word[i])  # Добавляем сам символ как токен
                    i += 1
                else:
                    # 2.3 Добавить токен в результат
                    tokens.append(find_token)
                    # 2.4 Увеличить i на длину токена
                    i += len(find_token)

        # 4. Заменить токены на их ID
        token_ids = self._tokens_to_ids(tokens)

        # Заменяем -1 на unk_token_id
        token_ids = [tid if tid != -1 else self.unk_token_id for tid in token_ids]

        # Добавляем специальные токены если нужно
        if add_special_tokens:
            if self.bos_token_id is not None:
                token_ids.insert(0, self.bos_token_id)
            if self.eos_token_id is not None:
                token_ids.append(self.eos_token_id)

        return token_ids

    def _find_max_matching_token(self, text: str, tokens: list) -> Optional[str]:
        """Находит самый длинный токен из списка, с которого начинается текст"""
        matching = [token for token in tokens if text.startswith(token)]
        return max(matching, key=len) if matching else None

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Конвертирует список токенов в их ID с обработкой неизвестных токенов"""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(-1)  # Специальное значение
        return ids

    def decode(self, tokens: List[int], **kwargs) -> str:
        """
        Декодирует последовательность токенов обратно в текстовую строку.

        Args:
        -----
        ids: List[int]
            Список токен-индексов для распаковки.

        Returns:
        --------
        text: str
            Оригинальный (или приближённый) раскодированный текст.

        Пример:
        -------
            >>> tokens = [15496, 11, 318, ...]
            >>> text = tokenizer.decode(tokens)
        """
        skip_special_tokens = kwargs.get("skip_special_tokens", True)

        # Поддержка torch.Tensor / numpy.ndarray наравне со списком int
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        # Фильтруем специальные токены если нужно
        if skip_special_tokens:
            tokens = [
                tid
                for tid in tokens
                if tid
                not in [
                    self.pad_token_id,
                    self.unk_token_id,
                    self.bos_token_id,
                    self.eos_token_id,
                ]
            ]

        # Конвертируем ID в токены
        token_strings = self._ids_to_tokens(tokens)

        # Объединяем токены в текст
        return "".join(token_strings)

    def _ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Конвертирует список Ids в их tokens"""
        tokens = []
        for token_id in ids:
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
            else:
                tokens.append(self.unk_token)  # Специальное значение
        return tokens

    def save(self, filepath: str):
        """
        Сохраняет токенизатор в файл.

        Args:
            filepath: Путь для сохранения
        """
        import json

        # Пары слияний в порядке ранга: список [левый, правый] переживает
        # JSON без потерь, в отличие от ключей "a,b" (токен может содержать запятую)
        merges_serializable = [
            list(pair) for pair, _ in sorted(self.merges.items(), key=lambda x: x[1])
        ]

        config = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "tokenizer_type": self.__class__.__name__,
            "merges": merges_serializable,
            "vocab_list": self.vocab_list,
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
            BPETokenizer: Загруженный токенизатор
        """
        import json

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
        tokenizer.vocab_list = config["vocab_list"]

        # Восстанавливаем пары слияний
        merges = config.get("merges", [])
        tokenizer.merges = {}
        if isinstance(merges, dict):
            # Старый формат: {"a,b": rank}
            for k, v in merges.items():
                parts = k.split(",")
                if len(parts) == 2:
                    tokenizer.merges[(parts[0], parts[1])] = v
        else:
            for rank, (left, right) in enumerate(merges):
                tokenizer.merges[(left, right)] = rank

        # Создаем обратный словарь
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

        # Обновляем ID специальных токенов
        tokenizer.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token)
        tokenizer.unk_token_id = tokenizer.vocab.get(tokenizer.unk_token)
        tokenizer.bos_token_id = tokenizer.vocab.get(tokenizer.bos_token)
        tokenizer.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token)

        return tokenizer


class SimpleBPETokenizer(BPETokenizer):
    """
    Упрощенная версия BPE токенизатора для демонстрации.
    Наследует вашу реализацию, но может быть упрощена при необходимости.
    """

    pass
