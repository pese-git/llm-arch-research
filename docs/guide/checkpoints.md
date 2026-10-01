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

**Голый `state_dict`.** Скрипт `run_llm_experiment.py` пока хранит веса (`torch.save(model.state_dict())`) и конфиг (JSON) отдельными файлами. Их загружают так:

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

Состояние оптимизатора и планировщика не сохраняется: `Trainer` на загруженной модели начнёт с нулевых моментов Adam и снова с warmup. Для настоящего продолжения обучения сохраняйте `trainer.optimizer.state_dict()` и `trainer.scheduler.state_dict()` в своём цикле.
