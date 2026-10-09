# Эксперименты с LLM-архитектурами

Скрипты обучения и генерации: напрямую через библиотеку `llm` (`llm_only/`) и через адаптер HuggingFace (`hf_integration/`).

Все команды запускаются **из корня репозитория**: пути в конфигах и скриптах (`checkpoints/...`) относительные.

## 📁 Структура

```
experiments/
├── llm_only/
│   ├── run_llm_experiment.py       # единый скрипт train/generate для всех 6 моделей
│   └── configs/
│       ├── <model>_train.json      # gpt, gpt2, llama, mistral, mixtral, gemma — учебный корпус
│       ├── <model>_generate.json
│       └── llama_corpus_*.json     # пример конфига с секцией data: корпус из файла
├── hf_integration/                 # только модель GPT (ограничение hf-proxy)
│   ├── test_hf_proxy.py            # smoke-тест адаптеров модели и токенизатора
│   ├── simple_hf_training.py       # ручной цикл обучения через hf-proxy
│   ├── train_with_hf_trainer.py    # обучение через transformers.Trainer
│   └── generate_with_hf_tools.py   # генерация через HF-интерфейсы
└── shared/
    ├── configs.py                  # учебный корпус TRAIN_TEXTS, пути PATHS, конфиги GPT для hf_integration
    ├── data.py                     # разбиение корпуса, ExperimentLogger, вспомогательные функции
    └── prepare_corpus.py           # текст → data/<name>/{train.bin, val.bin, tokenizer.json}
```

## 🚀 llm_only: обучение и генерация без HuggingFace

```bash
# Обучение
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action train --config experiments/llm_only/configs/llama_train.json

# Генерация обученной моделью
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action generate --config experiments/llm_only/configs/llama_generate.json
```

Аргументы:

| Аргумент | Значения |
|---|---|
| `--model`, `-m` | `gpt`, `gpt2`, `llama`, `mistral`, `mixtral`, `gemma` |
| `--action`, `-a` | `train`, `generate` |
| `--config`, `-c` | путь к JSON-конфигу |

**Что делает `train`:**
1. Без секции `data` берёт учебный корпус `TRAIN_TEXTS` из `shared/configs.py` (80% — train; валидационная часть не используется) и загружает BPE-токенизатор из `bpe_tokenizer` или обучает новый и сохраняет его туда. С секцией `data` берёт токенизатор и файлы токенов из неё (см. [Корпус из файла](#корпус-из-файла)).
2. Подставляет `vocab_size` токенизатора в `model_config`, создаёт модель и обучает её `llm.training.Trainer`; с `data.val` после каждой эпохи печатается валидационный loss, в конце — перплексия.
3. Сохраняет веса в `model_weights`, итоговый конфиг модели — в `model_config_path`, логи — в `log_path`.

**Что делает `generate`:** загружает токенизатор, конфиг и веса по путям из конфига и генерирует продолжение для каждого из `test_prompts`.

### Формат конфига обучения

```json
{
  "bpe_tokenizer": "checkpoints/bpe_tokenizer.json",
  "bpe_vocab_size": 1000,
  "bpe_special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
  "test_prompts": ["Машинное обучение"],
  "model_config": { "vocab_size": null, "embed_dim": 256, "...": "ключи зависят от модели" },
  "model_weights": "checkpoints/llama-bpe/model.pt",
  "model_config_path": "checkpoints/llama-bpe/config.json",
  "training": { "learning_rate": 0.0003, "batch_size": 2, "num_epochs": 3, "warmup_ratio": 0.1 },
  "log_path": "checkpoints/llama_only_training_logs.json"
}
```

Ключи `training` повторяют аргументы `Trainer` ([руководство](../docs/guide/training.md#trainer)): обязательные `learning_rate`, `batch_size`, `num_epochs` или `max_steps`, `warmup_ratio` или `warmup_steps`; необязательные `device` (`"auto"` — cuda, mps или cpu), `eval_interval`, `eval_batches`, `checkpoint_dir`, `save_interval`, `keep_best`, `seed`, `train_log_path` (JSON с loss и lr по шагам). `--resume checkpoints/<run>/last.pt` продолжает обучение с чекпоинта — конфиг должен быть тем же. Модель сохраняется через `model.save` (класс, конфиг и веса в одном файле); `generate` читает и старые файлы с голым `state_dict`.

Какие ключи `model_config` нужны каждой модели — см. [llm/README.md](../llm/README.md#ключи-конфига). Лишние ключи игнорируются: например, `num_experts`/`top_k_experts`/`window_size` в конфиге Gemma ни на что не влияют (`num_kv_heads` Gemma читает: по умолчанию 1 — MQA). `window_size` в Mistral и Mixtral необязателен: без него окна нет (как в Mixtral 8x7B, поэтому в `mixtral_train.json` его нет).

Все конфиги используют общий токенизатор `checkpoints/bpe_tokenizer.json`: если он уже есть, `bpe_vocab_size` и `bpe_special_tokens` не применяются.

### Корпус из файла

Для обучения на реальном корпусе текст один раз токенизируется в файлы `.bin`, которые читаются блоками без паддинга (`TokenBlockDataset`, см. [руководство](../docs/guide/data.md#корпус-из-файла)):

```bash
# текст UTF-8, пустая строка разделяет документы; --url вместо --input скачивает файл
uv run python experiments/shared/prepare_corpus.py --input corpus.txt --out data/corpus --vocab-size 8000

uv run python experiments/llm_only/run_llm_experiment.py --model llama --action train --config experiments/llm_only/configs/llama_corpus_train.json
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action generate --config experiments/llm_only/configs/llama_corpus_generate.json
```

`prepare_corpus.py` делит строки на train и val по хвосту файла (`--val-ratio`, 1 % по умолчанию), обучает BPE на первых `--tokenizer-lines` строках (или берёт готовый `--tokenizer`) и пишет `train.bin`, `val.bin`, `tokenizer.json` в `--out`. Каталог `data/` в `.gitignore`.

Конфиг обучения вместо `bpe_*` содержит секцию `data`; `block_size` блоков равен `max_position_embeddings` модели:

```json
{
  "data": { "train": "data/corpus/train.bin", "val": "data/corpus/val.bin", "tokenizer": "data/corpus/tokenizer.json" },
  "model_config": { "vocab_size": null, "embed_dim": 384, "num_heads": 6, "num_layers": 6, "max_position_embeddings": 256, "dropout": 0.0 },
  "model_weights": "checkpoints/llama-corpus/model.pt",
  "model_config_path": "checkpoints/llama-corpus/config.json",
  "training": { "learning_rate": 0.0006, "batch_size": 16, "device": "auto", "max_steps": 2000, "warmup_ratio": 0.05,
                "eval_interval": 200, "eval_batches": 50, "checkpoint_dir": "checkpoints/llama-corpus",
                "save_interval": 200, "seed": 0, "train_log_path": "checkpoints/llama-corpus/log.json" },
  "log_path": "checkpoints/llama_corpus_training_logs.json"
}
```

Прерванное обучение продолжается с последнего чекпоинта:

```bash
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action train --config experiments/llm_only/configs/llama_corpus_train.json --resume checkpoints/llama-corpus/last.pt
```

В конфиге генерации токенизатор указывается так же: `"data": { "tokenizer": "data/corpus/tokenizer.json" }` (или по-старому `bpe_tokenizer`).

### Формат конфига генерации

```json
{
  "bpe_tokenizer": "checkpoints/bpe_tokenizer.json",
  "test_prompts": ["Искусственный интеллект"],
  "model_config_path": "checkpoints/mistral-bpe/config.json",
  "model_weights": "checkpoints/mistral-bpe/model.pt",
  "generation": { "max_new_tokens": 40, "temperature": 0.8, "do_sample": true, "top_k": null, "top_p": null },
  "log_path": "checkpoints/mistral_only_generation_logs.json"
}
```

## 🤗 hf_integration: через hf-proxy

Работает только с моделью `GPT` — см. [hf-proxy/README.md](../hf-proxy/README.md).

```bash
# Smoke-тест адаптеров
uv run python experiments/hf_integration/test_hf_proxy.py

# Ручное обучение через hf-proxy → checkpoints/hf_simple_trained, checkpoints/hf_simple_tokenizer
uv run python experiments/hf_integration/simple_hf_training.py

# Генерация моделью из simple_hf_training.py (запускать после него)
uv run python experiments/hf_integration/generate_with_hf_tools.py

# Обучение через transformers.Trainer → checkpoints/hf-trained, checkpoints/hf-trained-proxy
uv run python experiments/hf_integration/train_with_hf_trainer.py
```

Конфиги этих скриптов (`BASE_GPT_CONFIG`, `BPE_CONFIG`, `TRAINING_CONFIG`, `GENERATION_CONFIG`, `PATHS`) задаются в `shared/configs.py`.

## 📊 Сравнение подходов

| Аспект | llm_only | hf_integration |
|---|---|---|
| Модели | все 6 | только GPT |
| Зависимости | PyTorch | + Transformers, Datasets |
| Обучение | `llm.training.Trainer` | ручной цикл или `transformers.Trainer` |
| Конфигурация | JSON-файлы в `llm_only/configs/` | Python-словари в `shared/configs.py` |

## 🛠️ Добавление эксперимента

- **Новая модель в llm_only:** добавьте ветку в `load_model_class()` в `run_llm_experiment.py` и пару конфигов `<model>_train.json` / `<model>_generate.json`.
- **Новый скрипт:** положите его в `llm_only/` или `hf_integration/`, используйте утилиты из `shared/` и сохраняйте результаты в `checkpoints/`.

## 📚 См. также

- [Учебное пособие по архитектурам](../docs/textbook/README.md)
- [Библиотека llm](../llm/README.md)
- [hf-proxy](../hf-proxy/README.md)
- [Ноутбуки](../notebooks/)
