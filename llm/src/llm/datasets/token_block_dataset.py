"""
Датасет непрерывных блоков токенов — для обучения на корпусе из файла.

Корпус — один плоский массив токенов (файл, записанный `tokenize_file`, или массив
в памяти). Пример i — токены с i·block_size по i·block_size + block_size: блоки
не пересекаются, паддинга нет, остаток короче блока отбрасывается. Так режут
непрерывный поток текста в GPT-2 и nanoGPT: каждая позиция блока — настоящий токен,
и вычисления не тратятся на паддинг, в отличие от TextDataset, где одна строка —
один пример.

Файл читается через numpy.memmap: в память попадают только запрошенные блоки,
поэтому размер корпуса ограничен диском, а не RAM. Перемешивание блоков — задача
DataLoader(shuffle=True); датасет детерминирован.

Формат файла: массив uint16 (словарь до 65536 токенов) или uint32 без заголовка;
dtype берётся из файла метаданных `<path>.json`, который пишет `tokenize_file`,
либо из аргумента dtype.
"""

import json
import os
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def read_token_file_meta(path: str) -> Optional[dict]:
    """
    Метаданные файла токенов: `<path>.json` (пишет `tokenize_file`) или None, если его нет.
    """
    meta_path = f"{path}.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


class TokenBlockDataset(Dataset):
    """
    Непрерывные блоки токенов фиксированной длины из файла или массива.

    Args:
        tokens: путь к файлу токенов (читается через numpy.memmap) либо одномерный
            массив или список токенов.
        block_size: длина блока — обычно `max_position_embeddings` модели.
        dtype: dtype файла (`"uint16"`, `"uint32"`); для файла без `<path>.json`
            обязателен, для массива игнорируется.

    Возвращает словарь с двумя long-тензорами формы [block_size]:
        - input_ids: токены блока;
        - labels: те же токены. Сдвиг на одну позицию делает loss
          (`llm.training.loss.causal_lm_loss`), как и для остальных датасетов.
    `attention_mask` нет: паддинга в блоках не бывает.

    Пример:
        >>> dataset = TokenBlockDataset("data/wiki/train.bin", block_size=256)
        >>> loader = DataLoader(dataset, batch_size=8, shuffle=True)
        >>> batch = next(iter(loader))
        >>> batch["input_ids"].shape
        torch.Size([8, 256])
    """

    def __init__(
        self,
        tokens: Union[str, os.PathLike, np.ndarray, Sequence[int]],
        block_size: int,
        *,
        dtype: Optional[str] = None,
    ):
        if block_size < 1:
            raise ValueError(f"block_size должен быть ≥ 1, получено {block_size}")
        self.block_size = block_size

        if isinstance(tokens, (str, os.PathLike)):
            path = os.fspath(tokens)
            if dtype is None:
                meta = read_token_file_meta(path)
                if meta is None or "dtype" not in meta:
                    raise ValueError(
                        f"Не найден {path}.json с dtype файла токенов: "
                        "запишите файл через tokenize_file или передайте dtype явно"
                    )
                dtype = meta["dtype"]
            self._tokens = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        else:
            self._tokens = np.asarray(tokens)
            if self._tokens.ndim != 1:
                raise ValueError(
                    f"Ожидался одномерный массив токенов, получена форма {self._tokens.shape}"
                )

        self.num_tokens = int(self._tokens.shape[0])
        if self.num_tokens < block_size:
            raise ValueError(
                f"В корпусе {self.num_tokens} токенов — меньше block_size = {block_size}: "
                "ни одного полного блока"
            )

    def __len__(self) -> int:
        """Число полных блоков; остаток короче block_size отбрасывается."""
        return self.num_tokens // self.block_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Индекс {idx} вне диапазона [0, {len(self)})")
        start = idx * self.block_size
        # Копия среза memmap в int64: тензор не должен ссылаться на файл
        chunk = np.array(self._tokens[start : start + self.block_size], dtype=np.int64)
        input_ids = torch.from_numpy(chunk)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
