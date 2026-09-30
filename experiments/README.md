# Эксперименты с LLM-архитектурами

Скрипты обучения и генерации: напрямую через библиотеку `llm` (`llm_only/`) и через адаптер HuggingFace (`hf_integration/`).

Все команды запускаются **из корня репозитория**: пути в конфигах и скриптах (`checkpoints/...`) относительные.

## 📁 Структура

```
experiments/
├── llm_only/
│   ├── run_llm_experiment.py       # единый скрипт train/generate для всех 6 моделей
│   └── configs/
│       ├── <model>_train.json      # gpt, gpt2, llama, mistral, mixtral, gemma
│       └── <model>_generate.json
├── hf_integration/                 # только модель GPT (ограничение hf-proxy)
│   ├── test_hf_proxy.py            # smoke-тест адаптеров модели и токенизатора
│   ├── simple_hf_training.py       # ручной цикл обучения через hf-proxy
│   ├── train_with_hf_trainer.py    # обучение через transformers.Trainer
│   └── generate_with_hf_tools.py   # генерация через HF-интерфейсы
└── shared/
    ├── configs.py                  # учебный корпус TRAIN_TEXTS, пути PATHS, конфиги GPT для hf_integration
    └── data.py                     # разбиение корпуса, ExperimentLogger, вспомогательные функции
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
1. Берёт учебный корпус `TRAIN_TEXTS` из `shared/configs.py` (80% — train; валидационная часть сейчас не используется).
2. Загружает BPE-токенизатор из `bpe_tokenizer` или обучает новый и сохраняет его туда.
3. Подставляет `vocab_size` токенизатора в `model_config`, создаёт модель и обучает её `llm.training.Trainer`.
4. Сохраняет веса в `model_weights`, итоговый конфиг модели — в `model_config_path`, логи — в `log_path`.

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

Какие ключи `model_config` нужны каждой модели — см. [llm/README.md](../llm/README.md#ключи-конфига). Лишние ключи игнорируются: например, `num_experts`/`top_k_experts`/`window_size` в конфиге Gemma ни на что не влияют (`num_kv_heads` Gemma читает: по умолчанию 1 — MQA). `window_size` в Mistral и Mixtral необязателен: без него окна нет (как в Mixtral 8x7B, поэтому в `mixtral_train.json` его нет).

Все конфиги используют общий токенизатор `checkpoints/bpe_tokenizer.json`: если он уже есть, `bpe_vocab_size` и `bpe_special_tokens` не применяются.

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
