import torch
from torch.utils.data import Dataset
from typing import List, Any


class StreamingTextDataset(Dataset):
    """
    StreamingTextDataset — потоковый датасет для LLM/NLP на базе списка строк.

    Назначение:
    -----------
    - Позволяет эффективно обрабатывать большие текстовые выборки, итерируя по заранее подготовленному списку строк.
    - При итерации строки токенизируются на лету, превращаются в примеры фиксированной длины block_size (padding/truncation внутри класса).
    - Поддерживает стандартный DataLoader PyTorch.

    Ключевые особенности:
    ---------------------
    - Не требует загрузки всей коллекции токенов в RAM: поддерживает работу с любым размером датасета, если список строк заранее подготовлен.
    - Каждый пример (sample) формируется при обращении; не хранит массив батчей, не использует файлы внутри.
    - Поддерживает любой токенизатор с методом encode (например, BPE, SentencePiece, HF Tokenizer).
    - batch_size и параллелизм (num_workers) контролируются через DataLoader.

    Аргументы конструктора:
    -----------------------
    texts: List[str] — список строк (предварительно загруженных обучающих примеров).
    tokenizer: BaseTokenizer/Any — объект с методом encode(str, **kwargs) -> List[int].
    block_size: int — длина одного выходного примера в токенах (padding/truncation если нужно).

    Пример использования:
    ---------------------
        >>> texts = open("wiki_sample.txt", encoding="utf-8").read().splitlines()
        >>> ds = StreamingTextDataset(texts, tokenizer=tokenizer, block_size=512)
        >>> loader = torch.utils.data.DataLoader(ds, batch_size=8)
        >>> for batch in loader:
        ...     print(batch['input_ids'].shape)  # torch.Size([8, 512])

    Особенности:
    ------------
    - Проектирован для бесконечного стриминга текстовых данных из больших коллекций.
    - При batch_size > 1 каждый batch формируется DataLoader-ом из yield'ов этого датасета.
    - Не работает с файлами напрямую, только со строками (списком).
    - Подходит для обучения LLM, тестирования, дообучения, оценки на больших потоковых данных.

    References:
    -----------
    - PyTorch IterableDataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    - HuggingFace streaming datasets: https://huggingface.co/docs/datasets/stream
    - Практика масштабного обучения LLM: https://github.com/karpathy/nanoGPT/issues/182
    """

    def __init__(self, texts: List[str], tokenizer: Any, block_size: int = 128):
        """
        Инициализация StreamingTextDataset из списка строк.

        Аргументы:
            texts (List[str]): Список строк — текстовые обучающие примеры; весь датасет должен помещаться в этот список.
            tokenizer (Any): Токенизатор с методом encode(text, **kwargs) -> List[int].
            block_size (int, по умолчанию 128): Желаемая длина токенизированного примера (padding/truncation внутри класса).

        Особенности:
            - Поддерживает итеративную загрузку, эффективен для больших текстовых выборок.
            - Каждый пример автоматически дополняется или усекается до block_size.
            - Не читает данные из файла/буфера, а только из заранее подготовленного списка строк.

        Пример:
            >>> ds = StreamingTextDataset(texts=all_lines, tokenizer=tokenizer, block_size=256)
            >>> for ex in ds:
            ...     print(ex['input_ids'].shape)  # torch.Size([256])
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Получаем pad_token_id из токенизатора
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    def __len__(self):
        """
        Возвращает количество доступных примеров в датасете.
    
        Returns:
            int: Число примеров (равно длине исходного списка строк).
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Получить обработанный пример по индексу из потокового датасета.

        Аргументы:
            idx (int): Индекс примера в исходном списке строк.

        Возвращает:
            dict: Словарь с тензорами для обучения LLM:
                - 'input_ids': torch.Tensor формы [block_size] — индексы токенов (padding/truncation выполнены)
                - 'labels': torch.Tensor формы [block_size] — целевые метки (обычно совпадают с input_ids)

        Пример:
            >>> item = dataset[10]
            >>> assert isinstance(item, dict)
            >>> assert item['input_ids'].shape == (block_size,)
            >>> assert 'labels' in item
        """
        text = self.texts[idx]

        # Токенизация на лету
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Обрезаем или дополняем до нужной длины
        if len(input_ids) > self.block_size:
            input_ids = input_ids[: self.block_size]
        else:
            input_ids = input_ids + [self.pad_token_id] * (
                self.block_size - len(input_ids)
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}