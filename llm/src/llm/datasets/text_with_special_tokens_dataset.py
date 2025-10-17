import torch
from torch.utils.data import Dataset
from typing import List, Any
from llm.datasets.text_dataset import TextDataset


class TextWithSpecialTokensDataset(TextDataset):
    """
    TextWithSpecialTokensDataset — датасет для языковых моделей с поддержкой специальных токенов (BOS, EOS, PAD).

    Назначение:
    -----------
    - Работает с уже готовым списком строк (не с файлом!).
    - Токенизирует строки с помощью заданного токенизатора, вручную вставляет специальные токены (BOS/ EOS/ PAD).
    - Обрезает или дополняет каждую последовательность до длины block_size.

    Аргументы конструктора:
    -----------------------
    texts (List[str]): Список обучающих строк (примеров).
    tokenizer (Any): Любой токенизатор с методом encode(text, **kwargs).
    block_size (int, default=128): Желаемая длина примера (padding/truncation).
    add_bos (bool, default=False): Если True, добавляет BOS-токен в начало каждой последовательности.
    add_eos (bool, default=False): Если True, добавляет EOS-токен в конец.

    Особенности:
    ------------
    - Если pad_token_id не задан — по умолчанию паддит нулями.
    - Все returned примеры — dict с 'input_ids' и 'labels' (shape == block_size).
    - Обрезание/дополнение учётное: BOS/EOS не "выдавливаются" обрезкой.
    - Пример вызова:
        >>> texts = ["пример текста", "ещё текст"]
        >>> ds = TextWithSpecialTokensDataset(texts, tokenizer, block_size=16, add_bos=True, add_eos=True)
        >>> out = ds[0]
        >>> assert out['input_ids'].shape == (16,)

    References:
    -----------
    - OpenAI GPT-2 data loader: https://github.com/openai/gpt-2/blob/master/src/encode.py
    - HuggingFace data docs: https://huggingface.co/docs/transformers/pad_truncation
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        block_size: int = 128,
        add_bos: bool = False,
        add_eos: bool = False,
    ):
        """
        Инициализация датасета с поддержкой специальных токенов.

        Args:
            texts (List[str]): Список строк (все ваши обучающие примеры).
            tokenizer (Any): Токенизатор с методом encode(text, **kwargs).
            block_size (int): Длина выходного примера.
            add_bos (bool): Добавлять ли BOS токен в начало.
            add_eos (bool): Добавлять ли EOS токен в конец.
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.add_bos = add_bos
        self.add_eos = add_eos

        for text in texts:
            # Кодируем с специальными токенами
            input_ids = tokenizer.encode(
                text, add_special_tokens=True, add_bos_token=add_bos, add_eos_token=add_eos
            )

            # Учитываем специальные токены при обрезке/дополнении
            effective_block_size = block_size
            if add_bos:
                effective_block_size -= 1
            if add_eos:
                effective_block_size -= 1

            if len(input_ids) > effective_block_size:
                input_ids = input_ids[:effective_block_size]

            # Добавляем специальные токены если нужно
            if (
                add_bos
                and hasattr(tokenizer, "bos_token_id")
                and tokenizer.bos_token_id is not None
            ):
                input_ids = [tokenizer.bos_token_id] + input_ids
            if (
                add_eos
                and hasattr(tokenizer, "eos_token_id")
                and tokenizer.eos_token_id is not None
            ):
                input_ids = input_ids + [tokenizer.eos_token_id]

            # Дополняем до полной длины
            pad_token_id = getattr(tokenizer, "pad_token_id", 0)
            if len(input_ids) < block_size:
                input_ids = input_ids + [pad_token_id] * (block_size - len(input_ids))

            self.examples.append(input_ids)

    def __len__(self):
        """
        Возвращает количество примеров в датасете.

        Returns:
            int: Размер (len(self.examples)).
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Получить пример с учётом специальных токенов и паддинга.

        Args:
            idx (int): Индекс в dataset.

        Returns:
            dict: {'input_ids': torch.Tensor [block_size], 'labels': torch.Tensor [block_size]}
        """
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}
