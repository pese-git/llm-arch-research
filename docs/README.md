# Документация архитектур

Разбор каждой реализованной в [`llm/`](../llm/src/llm) архитектуры: из чего состоит блок декодера, какие классы за что отвечают, какие параметры конфига на что влияют и что изменилось по сравнению с предыдущей моделью в линейке.

| Архитектура | Год / источник | Ключевые механизмы |
|---|---|---|
| [GPT-1](gpt.md) | OpenAI, 2018 | абсолютные позиционные эмбеддинги, стандартный MHA, **post-LN** |
| [GPT-2](gpt2.md) | OpenAI, 2019 | то же + переход на **pre-LN**, финальная нормализация |
| [LLaMA](llama.md) | Meta, 2023 | RoPE, RMSNorm, SwiGLU, обычный MHA (без GQA) |
| [Mistral](mistral.md) | Mistral AI, 2023 | + Grouped Query Attention, Sliding Window Attention |
| [Mixtral](mixtral.md) | Mistral AI, 2023 | Mistral + Mixture-of-Experts вместо плотного FFN |
| [Gemma](gemma.md) | Google DeepMind, 2024 | RoPE, RMSNorm, Multi-Query Attention, GeGLU |

Цепочка развития (кроме Gemma, которая — параллельная ветка на той же базе RoPE+RMSNorm): GPT-1 → GPT-2 → LLaMA → Mistral → Mixtral.

Общие механизмы, которые разные модели сочетают по-своему: [Attention и его виды](attention.md) — MHA, GQA, MQA, скользящее окно, KV-кэш; [маски](#маски) — ниже.

## Маски

Маска в attention указывает, на какие позиции токен может смотреть: перед `softmax` запрещённым парам (строка — запрос `i`, столбец — ключ `j`) в матрицу `scores` записывается `−∞`, и их веса становятся нулевыми. В репозитории три вида масок.

| Маска | Откуда | Где | Что запрещает |
|---|---|---|---|
| **Causal** | строится внутри attention (`_tril_mask`) | все модели | смотреть в будущее: `j > i` |
| **Скользящее окно** | строится внутри `GroupedQueryAttention`, если задан `window_size` | Mistral, Mixtral | то же и слишком далёкое прошлое: `i − j > window_size` |
| **`attention_mask`** | передаётся снаружи, `[batch, seq_len]`, 1 — токен, 0 — паддинг | параметр `forward` и `generate` всех моделей | смотреть на pad-токены |

### Causal-маска и скользящее окно

Causal-маска нужна для обучения на предсказании следующего токена: без неё, предсказывая токен `i + 1`, модель видела бы его во входе. Она накладывается всегда, в том числе с KV-кэшем. Маска хранится для абсолютных позиций `0 … max_seq_len − 1`, и при кэше берётся её срез: строки — новые токены `start_pos … start_pos + T − 1`, столбцы — все ключи, из кэша и новые. При генерации по одному токену строка одна и видит всё прошлое, но при нескольких новых токенах (префилл промпта кусками) срез не даёт им видеть друг друга «вперёд».

Скользящее окно Mistral/Mixtral дополнительно разрешает только `0 ≤ i − j ≤ window_size` — `window_size + 1` позиций вместе с самим токеном (почему `+ 1` — в [mistral.md](mistral.md#ширина-окна-w--1)). Кэш там хранит только последние `window_size` ключей, поэтому столбцы среза начинаются с позиции `start_pos − длина кэша`.

### `attention_mask` и паддинг

Последовательности в батче должны быть одной длины, и короткие дополняют pad-токенами. `attention_mask` отмечает, где настоящие токены, а где паддинг:

```
input_ids            attention_mask
Привет мир !         1 1 1
Да   <pad> <pad>     1 0 0      ← правый паддинг
<pad> <pad> Да       0 0 1      ← левый паддинг
```

**Правый паддинг** стоит после настоящих токенов, и causal-маска и так не даёт им на него смотреть. Выход для настоящих токенов не зависит от паддинга, а loss на pad-позициях при обучении отключают метками `-100`. Так дополняет батчи коллатор [hf-proxy](../hf-proxy/README.md).

**Левый паддинг** нужен для генерации батчем: все строки должны кончаться в одной позиции, чтобы новый токен дописывался сразу после текста. Здесь одной causal-маски мало: настоящие токены видят паддинг слева. Кроме маски ключей нужен ещё и сдвиг позиций: без него `Да` получает позицию 2 вместо 0, а от позиции зависят позиционные эмбеддинги и RoPE. В HuggingFace для этого из маски вычисляются `position_ids`.

Что поддерживается сейчас:

| Маска | `forward` | `generate` |
|---|---|---|
| `None` или из одних единиц | ✅ | ✅ |
| правый паддинг (в каждой строке единицы, затем нули) | ✅ — результат совпадает с прогоном без паддинга | ❌ `NotImplementedError`: генерация продолжилась бы после паддинга |
| левый паддинг, нули в середине | ❌ `NotImplementedError` | ❌ `NotImplementedError` |
| нули вместе с кэшем | ❌ `NotImplementedError` | — |

Маски, которые модели не умеют применить, отклоняются, а не игнорируются: иначе результат был бы молча неверным. Модули attention внешнюю маску не получают — проверка (`check_attention_mask` в [`core/generation.py`](../llm/src/llm/core/generation.py)) делается в `forward` модели. Полная поддержка паддинга (маска ключей и сдвиг позиций) — в [бэклоге](backlog.md#56-нет-поддержки-левого-паддинга--p2), пункт 56.

## Известные ограничения

Общие для всех архитектур:

- **`attention_mask`: поддерживается только правый паддинг** в `forward`; левый паддинг и генерация батчем промптов разной длины не поддерживаются (`NotImplementedError`), см. [Маски](#attention_mask-и-паддинг).

Специфичные для архитектуры:

- **Mistral, Mixtral** — окно sliding window шириной `window_size + 1` позиций (как в тексте статьи и prefill эталонного кода), а в HuggingFace — `window_size`; при загрузке весов HF — `window_size = sliding_window − 1`, см. [mistral.md](mistral.md#ширина-окна-w--1).

Полный список технического долга с приоритетами и способами исправления — в [backlog.md](backlog.md).

## Литература

Все статьи, на которые ссылаются документы по архитектурам.

### Архитектуры моделей

- Radford, Narasimhan, Salimans, Sutskever. *Improving Language Understanding by Generative Pre-Training*. OpenAI, 2018. [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (на arXiv не публиковалась)
- Radford, Wu, Child, Luan, Amodei, Sutskever. *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (на arXiv не публиковалась)
- Touvron et al. *LLaMA: Open and Efficient Foundation Language Models*. 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — GQA в линейке LLaMA появляется здесь (модель 70B)
- Jiang et al. *Mistral 7B*. 2023. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Jiang et al. *Mixtral of Experts*. 2024. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Gemma Team. *Gemma: Open Models Based on Gemini Research and Technology*. 2024. [arXiv:2403.08295](https://arxiv.org/abs/2403.08295)

### Основа: трансформер

- Vaswani et al. *Attention Is All You Need*. 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Liu et al. *Generating Wikipedia by Summarizing Long Sequences*. 2018. [arXiv:1801.10198](https://arxiv.org/abs/1801.10198) — decoder-only трансформер, на который опирается GPT-1

### Позиционное кодирование

- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

### Нормализация

- Ba, Kiros, Hinton. *Layer Normalization*. 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang, Sennrich. *Root Mean Square Layer Normalization*. 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Xiong et al. *On Layer Normalization in the Transformer Architecture*. 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) — почему pre-LN обучается стабильнее post-LN

### Внимание

Обзор видов attention — в [attention.md](attention.md).

- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*. 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) — Multi-Query Attention
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer*. 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) — sliding window attention

### Feed-forward и активации

- Hendrycks, Gimpel. *Gaussian Error Linear Units (GELUs)*. 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Shazeer. *GLU Variants Improve Transformer*. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — SwiGLU и GeGLU

### Mixture-of-Experts

- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. 2017. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. 2021. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) — load-balancing loss для роутера

### Токенизация

- Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units*. 2015. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) — BPE-токенизация

## Диаграммы

Диаграммы в каждом документе — на Mermaid (рендерятся нативно на GitHub). Схема блока каждой модели устроена одинаково:

- сверху вниз: `token ids` → эмбеддинги → стек декодеров → финальная нормализация → `Linear` → `logits`;
- зелёная рамка — один блок декодера, повторяется `num_layers` раз; внутри показан путь одного блока, пунктир — residual-связи;
- **жирная обводка** — то, что изменилось по сравнению с предыдущей моделью в линейке;
- пунктирная стрелка от `logits` — шаг генерации (`softmax` и выбор токена выполняются в `generate()`, а не в `forward`).

Цвета: синий — эмбеддинги токенов и attention, фиолетовый — обучаемые позиционные эмбеддинги (GPT) и FFN, бирюзовый — RoPE, серый — нормализация, линейные слои и dropout.

RoPE нарисован сбоку от декодера с пунктирной стрелкой в attention: он не прибавляется к основному потоку, как позиционные эмбеддинги GPT, а поворачивает Q и K внутри attention каждого слоя. Подробно — в [llama.md](llama.md#attention-с-rope).

Подробные схемы Multi-Head Attention, одной головы attention и FFN — в [gpt.md](gpt.md#устройство-компонентов), головы attention с RoPE — в [llama.md](llama.md#attention-с-rope), схема маршрутизации MoE — в [mixtral.md](mixtral.md#moe-изнутри).
