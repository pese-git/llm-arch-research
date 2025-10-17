import torch
from torch.utils.data import Dataset
from typing import List, Any


class TextDataset(Dataset):
    """
    TextDataset — простой датасет для подачи обучающих токенов LLM (batch-режим или по одному примеру).

    Назначение:
    -----------
    - Хранит последовательности текста (каждую строку или пример) в виде списка строк.
    - При обращении сам токенизирует строку в последовательность индексов с помощью заданного токенизатора.
    - Каждый пример автоматически усекётся или будет дополнен до фиксированной длины block_size (padding — zeros).

    Формат и аргументы конструктора:
    -------------------------------
    texts: List[str]
        Список строк, каждая из которых рассматривается как отдельный обучающий пример.
    tokenizer: любой объект с методом encode(str, **kwargs) → List[int]
        Обеспечивает сопоставление строки списку токенов (например, BPE, HuggingFace, SentencePiece и др.).
    block_size: int, по умолчанию 128
        Желаемая длина выходной последовательности (padding/truncation внутри класса).

    Особенности:
    ------------
    - Класс не работает с файлами напрямую: данные передаются готовым списком строк.
    - При недостаточной длине пример дополняется паддингом (нулём или другим токеном, зависит от реализации).
    - Может возвращать dict с input_ids, labels и прочими ключами (см. реализацию в функции __getitem__).

    Пример использования:
    ---------------------
        >>> with open("dataset.txt", encoding="utf-8") as f:
        ...     texts = f.read().splitlines()
        >>> dataset = TextDataset(texts, tokenizer, block_size=256)
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=4)
        >>> for item in loader:
        ...     # item['input_ids'] для обучения LLM

    References:
    -----------
    - Torch Dataset: https://pytorch.org/docs/stable/data.html
    - Примеры LLM датасетов в open-source: https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/tokenize.py
    """

    def __init__(self, texts: List[str], tokenizer: Any, block_size: int = 128):
        """
        Инициализация датасета из списка строк.

        Аргументы:
            texts (List[str]): Список строк — каждый элемент отдельный обучающий пример.
            tokenizer (Any): Токенизатор с методом encode(str, **kwargs) -> List[int].
            block_size (int, по умолчанию 128): Желаемая длина результата —
                длинные последовательности будут усечены, короткие — дополнены паддингом (pad_token_id или 0).

        Особенности:
            - Строки не фильтруются и не изменяются внутри датасета.
            - Для PAD используется pad_token_id из токенизатора (если есть) либо 0.
            - Dict, возвращаемый __getitem__, содержит 'input_ids' и 'labels'.

        Пример:
            >>> dataset = TextDataset(["hello world", "test string"], tokenizer, block_size=16)
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        for text in texts:
            # Кодируем текст в токены
            input_ids = tokenizer.encode(text, add_special_tokens=False)

            # Обрезаем или дополняем до нужной длины
            if len(input_ids) > block_size:
                input_ids = input_ids[:block_size]
            else:
                # Дополняем pad_token_id
                pad_token_id = getattr(tokenizer, "pad_token_id", 0)
                input_ids = input_ids + [pad_token_id] * (block_size - len(input_ids))

            self.examples.append(input_ids)

    def __len__(self):
        """
        Возвращает количество примеров в датасете (длина списка текстов).
    
        Returns:
            int: Число примеров в датасете.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Получить пример из датасета по индексу.

        Аргументы:
            idx (int): Индекс примера.

        Возвращает:
            dict: Словарь с тензорами токенов для модели:
                - 'input_ids': torch.Tensor shape [block_size], индексы токенов для входа.
                - 'labels': torch.Tensor shape [block_size], метки для LM задачи (обычно совпадают с input_ids).

        Пример:
            >>> item = dataset[7]
            >>> assert isinstance(item, dict)
            >>> assert item['input_ids'].shape == (block_size,)
            >>> assert 'labels' in item
        """
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}