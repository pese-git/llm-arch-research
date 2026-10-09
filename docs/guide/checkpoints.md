# Сохранение и загрузка
<!-- description: Сохранение и загрузка моделей библиотеки llm: model.save, Model.load и файлы токенизатора. -->

[← Генерация](generation.md) · [Оглавление](README.md) · [Загрузка весов HuggingFace →](hf-weights.md)

## Модель

```python
from llm.models.mistral import Mistral

model = Mistral({"vocab_size": 1000, "embed_dim": 64, "num_q_heads": 4, "num_kv_heads": 2,
                 "num_layers": 2, "max_position_embeddings": 128, "dropout": 0.0})
model.save("mistral.pt")                              # класс, конфиг и веса в одном файле
model = Mistral.load("mistral.pt", device="cpu")      # конфиг передавать не нужно; "cuda" — на GPU
```

- `save` пишет словарь `{"model_class", "config", "state_dict"}` через `torch.save`. Папку он не создаёт: создайте её заранее (`os.makedirs(..., exist_ok=True)`).
- `load` — метод класса: создаёт модель по сохранённому конфигу, загружает веса, переносит на `device` (по умолчанию `"cpu"`) и возвращает её **в режиме eval**. Файл, сохранённый на GPU, загружается и без GPU.
- Файл читается с `weights_only=True`: при загрузке не выполняется произвольный код.
- Загрузка файла другой модели (`GPT.load` на файле `Llama`) или голого `state_dict` — `ValueError`.
- Маски attention и таблицы RoPE в файл не попадают: они вычисляются из конфига.

**Скрипт экспериментов** `run_llm_experiment.py` сохраняет модель через `model.save` (и отдельно JSON конфига — для чтения глазами), так что `Llama.load("checkpoints/llama-bpe/model.pt")` работает. Файлы, записанные скриптом до [#81](https://github.com/pese-git/llm-arch-research/pull/81), — голый `state_dict`; `generate` читает оба формата, а вручную старый файл загружают так:

```python
import json, torch
from llm.models.llama import Llama

config = json.load(open("checkpoints/llama-bpe/config.json"))
model = Llama(config)
model.load_state_dict(torch.load("checkpoints/llama-bpe/model.pt", weights_only=True))
model.eval()
```

**Совместимость.** Форма весов и состав слоёв ни в одной модели не менялись, поэтому старые чекпоинты загружаются новым кодом. Изменения, после которых модель с теми же весами даёт другой результат, перечислены в [CHANGELOG](../../CHANGELOG.md).

## Токенизатор

```python
tokenizer.save("bpe_tokenizer.json")
tokenizer = BPETokenizer.load("bpe_tokenizer.json")
```

JSON хранит словарь, список слияний в порядке ранга и имена специальных токенов. Храните токенизатор рядом с моделью: модель без своего токенизатора бесполезна.

Файлы старого формата без поля `merges` загружаются, но кодируют текст жадным поиском самого длинного токена словаря — по словарю порядок слияний не восстановить. Чтобы кодировать по слияниям, переобучите токенизатор на том же корпусе.

## Продолжение обучения

`Trainer` с `checkpoint_dir` пишет `last.pt` и `best.pt` (см. [Обучение](training.md#чекпоинты-и-продолжение)); `trainer.save_checkpoint(path)` делает то же вручную. Файл — один `torch.save`, надмножество формата `model.save`:

```python
{
  "model_class": "Llama", "config": {...}, "state_dict": {...},   # ровно как BaseModel.save
  "trainer": {
    "format_version": 1,
    "state": {"step": 1200, "epoch": 0, "step_in_epoch": 50, "best_val_loss": 2.91,
              "loss_history": [...], "log": [...]},
    "optimizer": ..., "scheduler": ...,        # state_dict() AdamW и LambdaLR
    "rng": {"torch": ..., "cuda": ...},        # генераторы случайных чисел
    "args": {"lr": ..., "batch_size": ..., "num_epochs": ..., "max_steps": ...,
             "warmup_steps": ..., "warmup_ratio": ...},
  },
}
```

Все значения — тензоры, числа, строки, списки и словари, файл читается с `weights_only=True`. `BaseModel.load(path)` игнорирует секцию `trainer` и возвращает модель; `trainer.resume(path)` восстанавливает всё и проверяет, что класс модели и `args` совпадают с текущими. Файл `model.save` без секции `trainer` для `resume` не годится — `ValueError`. Моменты AdamW удваивают размер файла относительно весов.
