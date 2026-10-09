"""
Однократная токенизация текстового файла в файл токенов для TokenBlockDataset.

Корпус токенизируется один раз и сохраняется плоским массивом uint16 или uint32
без заголовка; рядом пишется `<out_path>.json` с dtype и статистикой. Дальше
обучение читает файл через numpy.memmap, не держа корпус в памяти и не вызывая
токенизатор заново на каждой эпохе — символьный BPE на Python для этого слишком
медленный.

Текст читается построчно, каждая строка кодируется отдельно (перевод строки
теряется, как и в остальных датасетах). Пустая строка — граница документа: если
задан eos_token_id, он ставится после каждого документа, чтобы модель училась
заканчивать текст, а блоки TokenBlockDataset не склеивали два документа без шва.
"""

import json
import os
from typing import Any, Optional

import numpy as np

# uint16 вмещает словарь до 65536 токенов — хватает GPT-2 (50257) и LLaMA 2 (32000)
_DTYPE_LIMITS = {"uint16": 2**16, "uint32": 2**32}


def _vocab_size(tokenizer: Any) -> Optional[int]:
    get = getattr(tokenizer, "get_vocab_size", None)
    return int(get()) if callable(get) else None


def tokenize_file(
    text_path: str,
    tokenizer: Any,
    out_path: str,
    *,
    eos_token_id: Optional[int] = None,
    dtype: Optional[str] = None,
    chunk_lines: int = 10_000,
) -> int:
    """
    Токенизирует текстовый файл в `out_path` (массив токенов) и `out_path.json`.

    Args:
        text_path: текст в UTF-8; пустая строка разделяет документы.
        tokenizer: объект с `encode(text, add_special_tokens=False) -> list[int]`;
            если есть `get_vocab_size()`, по нему выбирается dtype.
        out_path: куда писать токены, обычно `.../train.bin`.
        eos_token_id: токен конца документа; None — не вставлять.
        dtype: `"uint16"` или `"uint32"`; по умолчанию uint16, если словарь
            не больше 65536, иначе uint32. Без `get_vocab_size` и dtype — uint32.
        chunk_lines: сколько строк токенизировать между записями на диск.

    Returns:
        Число записанных токенов.

    Raises:
        ValueError: неизвестный dtype или id токена не помещается в него.
    """
    vocab_size = _vocab_size(tokenizer)
    if dtype is None:
        dtype = "uint16" if vocab_size is not None and vocab_size <= _DTYPE_LIMITS["uint16"] else "uint32"
    if dtype not in _DTYPE_LIMITS:
        raise ValueError(f"dtype должен быть uint16 или uint32, получено {dtype!r}")
    limit = _DTYPE_LIMITS[dtype]

    num_tokens = 0
    num_documents = 0
    buffer: list = []
    in_document = False

    def flush(f):
        nonlocal buffer, num_tokens
        if not buffer:
            return
        arr = np.asarray(buffer, dtype=np.int64)
        if arr.min() < 0 or arr.max() >= limit:
            raise ValueError(
                f"id токена {int(arr.max()) if arr.max() >= limit else int(arr.min())} "
                f"не помещается в {dtype} (допустимо 0..{limit - 1})"
            )
        arr.astype(np.dtype(dtype)).tofile(f)
        num_tokens += len(buffer)
        buffer = []

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(text_path, encoding="utf-8") as src, open(out_path, "wb") as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.rstrip("\n")
            if line.strip():
                buffer.extend(tokenizer.encode(line, add_special_tokens=False))
                in_document = True
            elif in_document:
                # Конец документа: пустая строка после непустых
                if eos_token_id is not None:
                    buffer.append(eos_token_id)
                num_documents += 1
                in_document = False
            if line_no % chunk_lines == 0:
                flush(dst)
        if in_document:
            if eos_token_id is not None:
                buffer.append(eos_token_id)
            num_documents += 1
        flush(dst)

    meta = {
        "dtype": dtype,
        "num_tokens": num_tokens,
        "num_documents": num_documents,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
    }
    with open(f"{out_path}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return num_tokens
