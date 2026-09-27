# Документация архитектур

Разбор каждой реализованной в [`llm/`](../llm/src/llm) архитектуры: из чего состоит блок декодера, какие классы за что отвечают, какие параметры конфига на что влияют и что изменилось по сравнению с предыдущей моделью в линейке.

| Архитектура | Год / источник | Ключевые механизмы |
|---|---|---|
| [GPT-1](gpt.md) | OpenAI, 2018 | абсолютные позиционные эмбеддинги, стандартный MHA, **post-LN** |
| [GPT-2](gpt2.md) | OpenAI, 2019 | то же + переход на **pre-LN**, финальная нормализация |
| [LLaMA](llama.md) | Meta, 2023 | RoPE, RMSNorm, SwiGLU (⚠️ без GQA, вопреки докстрингу) |
| [Mistral](mistral.md) | Mistral AI, 2023 | + Grouped Query Attention, Sliding Window Attention |
| [Mixtral](mixtral.md) | Mistral AI, 2023 | Mistral + Mixture-of-Experts вместо плотного FFN |
| [Gemma](gemma.md) | Google DeepMind, 2024 | RoPE, RMSNorm, Multi-Query Attention, GeGLU |

Цепочка развития (кроме Gemma, которая — параллельная ветка на той же базе RoPE+RMSNorm): GPT-1 → GPT-2 → LLaMA → Mistral → Mixtral.

## Известные ограничения

Общие для всех архитектур:

- **`attention_mask` не используется.** `generate(..., attention_mask=...)` и `HFGPTAdapter.forward` принимают маску, но дальше не передают: в батчах с паддингом модель смотрит на pad-токены.
- **Генерация дальше `max_position_embeddings`.** Длина проверяется только при вызове `forward` без кэша. В моделях с RoPE (LLaMA, Mistral, Mixtral, Gemma) генерация с кэшем за пределы буфера cos/sin падает с непонятным `RuntimeError` при `reshape`.
- **При переданном `cache` causal-маска не накладывается.** Это корректно, пока на вход подаётся по одному новому токену (как в `generate`), но не для нескольких токенов с кэшем.
- **Интерфейс `BaseModel` расходится с моделями.** Базовый класс объявляет `forward(input_ids, attention_mask) -> Tensor` и `generate(input_ids, max_length)`, а модели реализуют `forward(x, use_cache, cache) -> (logits, cache)` и общую сигнатуру `generate`, описанную в [gpt.md](gpt.md#генерация).
- **Ключ `head_size` в конфигах не читается** ни одной моделью: размер головы всегда `embed_dim // <число голов>`.

Специфичные для архитектуры:

- **GPT, GPT-2** — при генерации с KV-кэшем позиционный эмбеддинг всегда берётся для позиции 0, см. [gpt.md](gpt.md#генерация).
- **Mistral, Mixtral** — KV-кэш со скользящим окном расходится с генерацией без кэша, см. [mistral.md](mistral.md#sliding-window-attention).
- **Mixtral** — MoE без load-balancing loss, см. [mixtral.md](mixtral.md#moe-изнутри).
- **LLaMA** — нет GQA, вопреки докстрингу, см. [llama.md](llama.md#известное-расхождение-с-докстрингом).
- **Gemma** — в конфиге есть ключи, которые не используются, см. [gemma.md](gemma.md#неиспользуемые-ключи-конфига).

## Диаграммы

Диаграммы в каждом документе — на Mermaid (рендерятся нативно на GitHub). Диаграммы GPT-1 также существуют как drawio-исходники в [`assets/drawio/`](../assets/drawio).
