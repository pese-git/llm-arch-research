# Gemma

> Реализация: [`llm/src/llm/models/gemma/gemma.py`](../llm/src/llm/models/gemma/gemma.py) · класс `Gemma`
> Ноутбук: [`notebooks/gemma.ipynb`](../notebooks/gemma.ipynb)

Место в линейке: развивает ту же базу (RoPE + RMSNorm), что и [LLaMA](llama.md)/[Mistral](mistral.md), но с собственным вариантом attention и FFN — не входит в основную цепочку GPT → Mixtral.

## Обзор

Gemma (Google DeepMind, 2024, [arXiv:2403.08295](https://arxiv.org/abs/2403.08295)) в этом репозитории реализована как RoPE + RMSNorm трансформер с **Multi-Query Attention** (MQA — одна общая голова K/V на все Q-головы, предельный случай GQA) и **GeGLU**-FFN (GELU-gated, а не SiLU-gated, как в SwiGLU).

## Архитектура блока декодера

```mermaid
%%{init: {"flowchart": {"rankSpacing": 28, "nodeSpacing": 28}}}%%
flowchart TB
    Ids(["token ids"]):::io --> TokEmb["Token Embedding"]:::blue
    TokEmb --> Drop["Dropout"]:::gray
    subgraph Dec["GemmaDecoder × num_layers · pre-RMSNorm"]
        direction TB
        X(["x"]):::io --> N1["RMSNorm"]:::gray
        N1 --> Attn["Multi-Query Attention<br/>1 общая K/V-голова"]:::blueHl
        R["RoPE<br/>cos/sin от позиции · без параметров<br/>один модуль на все слои"]:::rope
        R -. "поворот Q и K" .-> Attn
        Attn --> A1(("+")):::add
        X -. residual .-> A1
        A1 --> N2["RMSNorm"]:::gray
        N2 --> FFN["GeGLU"]:::purpleHl
        FFN --> A2(("+")):::add
        A1 -. residual .-> A2
    end
    Drop --> Dec
    Dec --> NF["RMSNorm<br/>(финальный)"]:::gray --> Lin
    Lin["Linear → vocab_size"]:::gray --> Out(["logits"]):::io
    Out -. "generate(): softmax → выбор токена" .-> Next(["следующий токен"]):::io
    style Dec fill:transparent,stroke:#82b366,stroke-width:2px,color:#5b9a3c

    classDef io fill:#ffffff,stroke:#999999,color:#1a1a1a;
    classDef add fill:#ffffff,stroke:#666666,color:#1a1a1a;
    classDef blue fill:#dae8fc,stroke:#6c8ebf,color:#1a1a1a;
    classDef blueHl fill:#dae8fc,stroke:#2f5f9e,stroke-width:3px,color:#1a1a1a;
    classDef purple fill:#e1d5e7,stroke:#9673a6,color:#1a1a1a;
    classDef purpleHl fill:#e1d5e7,stroke:#6a3d85,stroke-width:3px,color:#1a1a1a;
    classDef gray fill:#f5f5f5,stroke:#666666,color:#1a1a1a;
    classDef grayHl fill:#f5f5f5,stroke:#333333,stroke-width:3px,color:#1a1a1a;
    classDef gold fill:#fff2cc,stroke:#d6b656,color:#1a1a1a;
    classDef rope fill:#d5f0ec,stroke:#3a9e8f,color:#1a1a1a;
    classDef ropeHl fill:#d5f0ec,stroke:#1f6f63,stroke-width:3px,color:#1a1a1a;
    classDef dim fill:#f5f5f5,stroke:#bbbbbb,color:#999999,stroke-dasharray:4 3;
```

Как RoPE поворачивает Q и K — в разделе [Attention с RoPE](llama.md#attention-с-rope) документа LLaMA.

### Multi-Query Attention vs GQA

MQA предложена в [Shazeer, 2019](https://arxiv.org/abs/1911.02150), GQA — в [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) как обобщение между MQA и MHA. В [Mistral](mistral.md#grouped-query-attention) число KV-голов — настраиваемый параметр (`num_kv_heads`), обычно несколько. В реализации MQA здесь этого параметра вообще нет: `MultiQueryAttention` всегда использует **одну** общую K/V-голову на все Q-головы ([`core/multi_query_attention.py`](../llm/src/llm/core/multi_query_attention.py)) — это не частный случай настраиваемой GQA, а отдельный, более узкий механизм.

## Компоненты

| Компонент | Класс | Файл |
|---|---|---|
| Токен-эмбеддинги | `TokenEmbeddings` | [`core/token_embeddings.py`](../llm/src/llm/core/token_embeddings.py) |
| Позиционное кодирование | `RoPE` | [`core/rope.py`](../llm/src/llm/core/rope.py) |
| Нормализация | `RMSNorm` | [`core/rms_norm.py`](../llm/src/llm/core/rms_norm.py) |
| Attention | `MultiQueryAttention` (1 общая K/V-голова + RoPE) | [`core/multi_query_attention.py`](../llm/src/llm/core/multi_query_attention.py) |
| FFN | `GeGLU` (gated GELU-MLP) | [`core/geglu.py`](../llm/src/llm/core/geglu.py) |
| Блок декодера | `GemmaDecoder` (pre-LN) | [`core/gemma_decoder.py`](../llm/src/llm/core/gemma_decoder.py) |
| Модель целиком | `Gemma` | [`models/gemma/gemma.py`](../llm/src/llm/models/gemma/gemma.py) |

`GemmaDecoder.forward` — та же pre-LN схема:
```
norm1_out = RMSNorm1(x)
attn_out  = MQA(norm1_out)           # с RoPE
out       = attn_out + x
norm2_out = RMSNorm2(out)
ffn_out   = GeGLU(norm2_out)
result    = ffn_out + out
```

## Конфигурация

Пример из [`experiments/llm_only/configs/gemma_train.json`](../experiments/llm_only/configs/gemma_train.json):

| Параметр | Значение в примере | Используется? |
|---|---|---|
| `vocab_size` | (из токенизатора) | ✅ |
| `embed_dim` | 256 | ✅ |
| `num_q_heads` | 4 | ✅ (единственный параметр числа голов, который читает `Gemma.__init__`) |
| `num_layers` | 4 | ✅ |
| `max_position_embeddings` | 512 | ✅ |
| `dropout` | 0.1 | ✅ |
| `head_size` | 64 | ❌ не читается — вычисляется как `embed_dim // num_q_heads` |
| `num_kv_heads` | 2 | ❌ не читается |
| `num_experts` | 8 | ❌ не читается |
| `top_k_experts` | 2 | ❌ не читается |
| `window_size` | 16 | ❌ не читается |

## Неиспользуемые ключи конфига

`Gemma.__init__` ([`models/gemma/gemma.py`](../llm/src/llm/models/gemma/gemma.py)) передаёт в `GemmaDecoder` только `num_q_heads`, `emb_size`, `head_size`, `max_seq_len`, `rope`, `dropout`. Ключи `head_size`, `num_kv_heads`, `num_experts`, `top_k_experts`, `window_size`, присутствующие в [`gemma_generate.json`](../experiments/llm_only/configs/gemma_generate.json)/[`gemma_train.json`](../experiments/llm_only/configs/gemma_train.json) (судя по всему, скопированные из конфига Mixtral), моделью не используются и ни на что не влияют. Это не баг в смысле краша — конструктор просто их игнорирует, — но конфиг вводит в заблуждение: MoE и настраиваемый GQA в текущей реализации Gemma отсутствуют, там всегда MQA с ровно одной K/V-головой.

## Генерация

`Gemma.generate(...)` — унифицированная сигнатура (см. [gpt.md](gpt.md#генерация)).

## Литература

Основная статья:

- Gemma Team. *Gemma: Open Models Based on Gemini Research and Technology*. 2024. [arXiv:2403.08295](https://arxiv.org/abs/2403.08295)

Компоненты:

- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*. 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) — Multi-Query Attention
- Shazeer. *GLU Variants Improve Transformer*. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — SwiGLU и GeGLU
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Zhang, Sennrich. *Root Mean Square Layer Normalization*. 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
