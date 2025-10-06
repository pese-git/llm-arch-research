"""
BPE (Byte Pair Encoding) токенизатор.

Реализация алгоритма BPE для токенизации текста.
"""

from typing import List, Dict, Tuple, Optional
from .base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE токенизатор для обработки текста.

    Реализует алгоритм Byte Pair Encoding для создания субсловных токенов.
    Использует вашу реализацию BPE.

    Примеры использования:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train(["пример текста для обучения"], vocab_size=1000)
        >>> tokens = tokenizer.encode("новый текст")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(self):
        super().__init__()
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab_list: List[str] = []

    def train(self, texts: List[str], vocab_size: int = 1000, **kwargs):
        """
        Обучение BPE токенизатора на текстах.

        Args:
            texts: Список текстов для обучения
            vocab_size: Желаемый размер словаря
            **kwargs: Дополнительные параметры
                - special_tokens: Список специальных токенов
        """
        # Объединяем все тексты в одну строку для обучения
        combined_text = " ".join(texts)

        # 1. Получаем уникальные токены (символы)
        unique_tokens = sorted(set(combined_text))
        tokens = unique_tokens.copy()

        # 2. Разбиваем текст на токены-символы
        sequence = list(combined_text)

        # 3. Объединяем токены до достижения нужного размера словаря
        while len(tokens) < vocab_size:
            # Считаем частоты пар
            pair_freq = {}
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                if pair not in pair_freq:
                    pair_freq[pair] = 0
                pair_freq[pair] += 1

            if not pair_freq:
                break  # нет пар — выходим

            # Находим самую частую пару (в случае равенства — та, что встретилась первой)
            most_frequent_pair = max(
                pair_freq.items(),
                key=lambda x: (x[1], -self._pair_first_index(sequence, x[0])),
            )[0]

            # Создаем новый токен
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            tokens.append(new_token)

            i = 0
            new_sequence = []

            while i < len(sequence):
                if (
                    i < len(sequence) - 1
                    and (sequence[i], sequence[i + 1]) == most_frequent_pair
                ):
                    new_sequence.append(new_token)
                    i += 2  # пропускаем два символа — заменённую пару
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            sequence = new_sequence

        # 4. Создаем словари
        self.vocab_list = tokens.copy()
        self.vocab = dict(zip(tokens, range(vocab_size)))
        self.inverse_vocab = dict(zip(range(vocab_size), tokens))
        self.vocab_size = len(self.vocab)

        # Добавляем специальные токены если указаны
        special_tokens = kwargs.get(
            "special_tokens",
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token],
        )
        self.add_special_tokens(special_tokens)

    def _pair_first_index(self, sequence, pair):
        """Находит первый индекс пары в последовательности."""
        for i in range(len(sequence) - 1):
            if (sequence[i], sequence[i + 1]) == pair:
                return i
        return float("inf")  # если пара не найдена (в теории не должно случиться)

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

        # 1. Разбиваем текст на токены-символы
        sequence = list(text)
        # 2. Инициализация пустого списка токенов
        tokens = []
        # 3. Установить i = 0
        i = 0
        while i < len(text):
            # 3.1 Найти все токены в словаре, начинающиеся с text[i]
            start_char = text[i]
            result = [
                token for token in self.vocab_list if token.startswith(start_char)
            ]
            # 3.2 Выбрать самый длинный подходящий токен
            find_token = self._find_max_matching_token(text[i:], result)
            if find_token is None:
                # Обработка неизвестного символа
                tokens.append(text[i])  # Добавляем сам символ как токен
                i += 1
            else:
                # 3.3 Добавить токен в результат
                tokens.append(find_token)
                # 3.4 Увеличить i на длину токена
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
        Декодирует последовательность токенов в текст.

        Args:
            tokens: Список идентификаторов токенов
            **kwargs: Дополнительные параметры
                - skip_special_tokens: Пропускать специальные токены

        Returns:
            str: Декодированный текст
        """
        skip_special_tokens = kwargs.get("skip_special_tokens", True)

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

        # Преобразуем кортежи в строки для JSON сериализации
        merges_serializable = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

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

        # Восстанавливаем кортежи из строк
        tokenizer.merges = {}
        for k, v in config["merges"].items():
            parts = k.split(",")
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


class SimpleBPETokenizer(BPETokenizer):
    """
    Упрощенная версия BPE токенизатора для демонстрации.
    Наследует вашу реализацию, но может быть упрощена при необходимости.
    """

    pass
