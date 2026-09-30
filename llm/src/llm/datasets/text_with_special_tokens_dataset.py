import torch
from torch.utils.data import Dataset
from typing import List, Any
from llm.datasets.lm_example import lm_example
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
    - Все returned примеры — dict с 'input_ids', 'attention_mask' и 'labels' (shape == block_size);
      на pad-позициях labels = -100. BOS/EOS — настоящие токены: маска 1, входят в loss.
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
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)

        for text in texts:
            # Кодируем без специальных токенов: bos/eos добавляются ниже ровно
            # по одному. Иначе токенизатор (например, BPETokenizer при
            # add_special_tokens=True) добавил бы свои и они задвоились бы.
            input_ids = tokenizer.encode(text, add_special_tokens=False)

            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            use_bos = add_bos and bos_token_id is not None
            use_eos = add_eos and eos_token_id is not None

            # Оставляем место под специальные токены при обрезке
            effective_block_size = block_size - int(use_bos) - int(use_eos)
            if len(input_ids) > effective_block_size:
                input_ids = input_ids[:effective_block_size]

            if use_bos:
                input_ids = [bos_token_id] + input_ids
            if use_eos:
                input_ids = input_ids + [eos_token_id]

            # Храним без паддинга: дополняет lm_example в __getitem__
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
            dict: {'input_ids', 'attention_mask', 'labels'} — torch.Tensor [block_size];
                labels на pad-позициях — -100.
        """
        return lm_example(self.examples[idx], self.block_size, self.pad_token_id)
