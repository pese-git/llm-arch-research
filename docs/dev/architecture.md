# Устройство репозитория
<!-- description: Устройство репозитория: пакеты, модули llm, контракт BaseModel, поток данных в forward и generate. -->

[Оглавление](README.md) · [Добавление модели →](adding-model.md)

## Workspace

Репозиторий — [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): корневой `pyproject.toml` объединяет два пакета и закрепляет общие зависимости (`torch==2.8.0`).

| Каталог | Что это | Зависит от |
|---|---|---|
| `llm/` | основная библиотека: блоки трансформера, шесть моделей, токенизатор, датасеты, `Trainer` | `torch>=2.3`, `numpy` |
| `hf-proxy/` | адаптер к HuggingFace Transformers (только `GPT`) | `llm`, `transformers`, `datasets` |
| `experiments/` | скрипты обучения и генерации: `llm_only/` (все модели, JSON-конфиги), `hf_integration/` (через hf-proxy), `shared/` (учебный корпус, пути, утилиты) | оба пакета |
| `notebooks/` | пошаговый разбор каждой архитектуры и BPE | оба пакета |
| `docs/` | документация: `textbook/`, `guide/`, `dev/` | — |
| `site/` | сайт документации (Astro + Starlight), собирается из `docs/` | Node.js |

**Библиотека `llm` зависит только от PyTorch и NumPy** — не добавляйте в неё `transformers` и другие тяжёлые зависимости. Всё, что связано с HuggingFace, живёт в `hf-proxy` или в тестах (с `pytest.importorskip("transformers")`).

## Пакет llm

```
llm/src/llm/
├── core/          # строительные блоки и общая логика моделей
├── models/        # шесть моделей: gpt/ (GPT, GPT2), llama/, mistral/, mixtral/, gemma/
├── tokenizers/    # BaseTokenizer, BPETokenizer
├── datasets/      # TextDataset, StreamingTextDataset, TextWithSpecialTokensDataset, lm_example
├── training/      # Trainer, get_optimizer, get_linear_schedule_with_warmup
└── evaluation/    # заготовка, пока пуста
```

### core/

| Группа | Модули |
|---|---|
| Базовый класс и общая логика | `base_model.py` (`BaseModel`: `generate`, `save`/`load`), `generation.py` (проверки аргументов, выбор токена, окно контекста), `padding.py` (маска ключей и позиции из `attention_mask`), `config_checks.py` (`resolve_head_size`), `weight_init.py` |
| Эмбеддинги и позиции | `token_embeddings.py` (+ `output_projection` для weight tying), `positional_embeddings.py` (обучаемые, GPT), `rope.py` |
| Attention | `multi_head_attention.py` (MHA, опционально RoPE), `group_query_attention.py` (GQA + скользящее окно), `multi_query_attention.py` (MQA, учебный модуль — моделями не используется) |
| FFN и активации | `feed_forward.py` (GELU-FFN), `swi_glu.py`, `geglu.py`, `gelu.py`, `silu.py`, `moe.py` (+ `load_balancing_loss`) |
| Нормализация | `rms_norm.py` (LayerNorm — из PyTorch) |
| Блоки декодера | `gpt_decoder.py` (post-LN), `gpt2_decoder.py` (pre-LN), `cached_decoder.py` (параметризуемый pre-LN блок LLaMA), `mistral_decoder.py`, `mixtral_decoder.py`, `gemma_decoder.py` |

Каждая модель — это `models/<name>/<name>.py`: эмбеддинги, стек блоков из `core/`, финальная нормализация и выходная проекция. Перенос весов HF — `models/<name>/hf_weights.py` (у Mistral и Mixtral — реэкспорт функции LLaMA).

## Контракт BaseModel

Модель — наследник `BaseModel(config)`, который сохраняет `self.config` и даёт общие `generate`, `save`, `load`, `auxiliary_loss`, `max_seq_len`. Модель обязана:

- в `__init__` вызвать `super().__init__(config)`, записать `self._max_seq_len = config["max_position_embeddings"]`, построить слои и инициализировать веса (`init_normal_` с `initializer_range`);
- реализовать `forward(x, use_cache=False, cache=None, attention_mask=None) -> (logits, cache)`:
  - `start_pos = cache_start_pos(cache)` — сколько токенов уже в кэше;
  - `check_sequence_length(seq_len, start_pos, max_seq_len)` — длина с учётом кэша;
  - `padding = padding_from_attention_mask(attention_mask, x, start_pos)` — `None` без нулей в маске, иначе маска ключей и позиции, которые передаются в каждый блок и в attention;
  - вернуть `(logits, new_cache)` при `use_cache=True` и `(logits, None)` иначе;
- если у модели есть вспомогательный loss (роутер MoE), переопределить `auxiliary_loss()` — `Trainer` прибавит его к loss при обучении.

`BaseModel.generate` работает поверх этого `forward` и одинаков для всех моделей: проверяет аргументы (`validate_sampling_args`, `check_generation_mask`), на каждом шаге выбирает вход (`next_generation_input`: только новый токен с кэшем; последние `max_seq_len` токенов без кэша, когда текст длиннее контекста), наращивает `attention_mask` и выбирает токен (`sample_next_token`).

**Формат кэша** зависит от модуля attention: `MultiHeadAttention` хранит `(K, V)`, `GroupedQueryAttention` — `(K, V, next_pos)`, потому что кэш со скользящим окном обрезается и его длина перестаёт совпадать с позицией. `cache_start_pos` понимает оба формата.

**Буферы.** Маски (`_tril_mask`) и таблицы RoPE регистрируются с `persistent=False`: их нет в `state_dict`, они вычисляются из конфига. Один объект `RoPE` создаётся в модели и передаётся во все блоки.

## Документация и сайт

`docs/` — единственный источник текста: markdown под GitHub (формулы `` $`…`$ `` и ` ```math `, схемы Mermaid). `site/scripts/sync-docs.mjs` копирует `docs/**/*.md` в коллекцию Starlight и строит меню по разделам «Оглавление» в `docs/<раздел>/README.md`; главная страница сайта — визитка `site/src/landing/index.mdx` (на GitHub входная точка — `docs/README.md`); плагин `site/src/plugins/remark-github-docs.mjs` переводит формулы в KaTeX, ссылки на главы — в страницы сайта, ссылки на код — на GitHub. Workflow `.github/workflows/docs-site.yml` проверяет сборку в PR, где меняются `docs/` или `site/`; сайт публикуется Docker-образом на кластере. Подробнее — [site/README.md](../../site/README.md).
