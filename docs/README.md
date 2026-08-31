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

Диаграммы в каждом документе — на Mermaid (рендерятся нативно на GitHub). Диаграммы GPT-1 также существуют как drawio-исходники в [`assets/drawio/`](../assets/drawio).
