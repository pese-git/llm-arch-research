#!/usr/bin/env python3
"""
Подготовка корпуса для обучения: текст → train.bin, val.bin, tokenizer.json.

    uv run python experiments/shared/prepare_corpus.py --input corpus.txt --out data/wiki
    uv run python experiments/shared/prepare_corpus.py --url https://.../tinystories.txt --out data/tinystories

Что делает:
1. Берёт текстовый файл (`--input`) или скачивает его по `--url` в `<out>/corpus.txt`.
   Формат — UTF-8, пустая строка разделяет документы.
2. Делит строки на train и val по хвосту файла (`--val-ratio`, по умолчанию 1 %): хвост,
   а не случайная выборка, чтобы соседние строки одного документа не оказались по разные
   стороны разбиения. Пишет `<out>/train.txt` и `<out>/val.txt`.
3. Загружает BPETokenizer из `--tokenizer` или обучает новый на первых
   `--tokenizer-lines` строках train (символьный BPE на Python медленный, весь корпус
   ему не нужен) и сохраняет в `<out>/tokenizer.json`.
4. Токенизирует оба сплита в `<out>/train.bin` и `<out>/val.bin` с `eos` между документами
   (`llm.datasets.tokenize_corpus.tokenize_file`).

Пути в секции `data` конфига обучения указывают на эти файлы. Каталог `data/` в .gitignore.
"""

import argparse
import os
import sys
import time
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.datasets.tokenize_corpus import tokenize_file
from llm.tokenizers import BPETokenizer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def split_lines(src_path, train_path, val_path, val_ratio):
    """Пишет хвост файла длиной val_ratio строк в val, остальное в train; возвращает счётчики."""
    with open(src_path, encoding="utf-8") as f:
        lines = f.readlines()
    num_val = int(len(lines) * val_ratio)
    # Граница — ближайшая пустая строка после начала хвоста, чтобы не резать документ
    cut = len(lines) - num_val
    boundary = cut
    while 0 < boundary < len(lines) and lines[boundary - 1].strip():
        boundary += 1
    if boundary < len(lines):
        cut = boundary
    # Иначе в хвосте нет пустой строки (корпус из одного документа без разделителей): режем по
    # заданной доле, а не уводим границу в конец файла — там валидации не осталось бы совсем
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:cut])
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(lines[cut:])
    return cut, len(lines) - cut


def main():
    parser = argparse.ArgumentParser(description="Подготовка корпуса: текст → .bin + tokenizer.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="текстовый файл UTF-8")
    source.add_argument("--url", help="URL текстового файла; скачивается в <out>/corpus.txt")
    parser.add_argument("--out", required=True, help="каталог для train.bin, val.bin, tokenizer.json")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="доля строк в валидации (хвост файла)")
    parser.add_argument("--tokenizer", help="готовый tokenizer.json; без него BPE обучается заново")
    parser.add_argument("--vocab-size", type=int, default=8000, help="vocab_size для обучения BPE")
    parser.add_argument("--tokenizer-lines", type=int, default=20_000, help="сколько строк train брать для обучения BPE")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src = args.input
    if args.url:
        src = os.path.join(args.out, "corpus.txt")
        print(f"⬇️  Скачивание {args.url} → {src}")
        urllib.request.urlretrieve(args.url, src)

    train_txt = os.path.join(args.out, "train.txt")
    val_txt = os.path.join(args.out, "val.txt")
    n_train, n_val = split_lines(src, train_txt, val_txt, args.val_ratio)
    print(f"✂️  Строк: train {n_train}, val {n_val}")
    if n_val == 0:
        print("⚠️  Валидационная часть пуста: увеличьте --val-ratio, иначе обучение с секцией data.val не запустится")

    tokenizer_path = os.path.join(args.out, "tokenizer.json")
    if args.tokenizer:
        tokenizer = BPETokenizer.load(args.tokenizer)
        if os.path.abspath(args.tokenizer) != os.path.abspath(tokenizer_path):
            tokenizer.save(tokenizer_path)
        print(f"📝 Токенизатор загружен: {args.tokenizer} (vocab_size={tokenizer.get_vocab_size()})")
    else:
        with open(train_txt, encoding="utf-8") as f:
            sample = [line.rstrip("\n") for _, line in zip(range(args.tokenizer_lines), f) if line.strip()]
        print(f"🔧 Обучение BPE на {len(sample)} строках, vocab_size={args.vocab_size}...")
        t0 = time.time()
        tokenizer = BPETokenizer()
        tokenizer.train(texts=sample, vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.save(tokenizer_path)
        print(f"✅ Токенизатор обучен за {time.time() - t0:.0f} с: {tokenizer_path} (vocab_size={tokenizer.get_vocab_size()})")

    for name, txt in (("train", train_txt), ("val", val_txt)):
        out = os.path.join(args.out, f"{name}.bin")
        t0 = time.time()
        n = tokenize_file(txt, tokenizer, out, eos_token_id=tokenizer.eos_token_id)
        print(f"✅ {out}: {n} токенов за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
