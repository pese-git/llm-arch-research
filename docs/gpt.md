# GPT-1

> Реализация: [`llm/src/llm/models/gpt/gpt.py`](../llm/src/llm/models/gpt/gpt.py) · класс `GPT`
> Ноутбук: [`notebooks/gpt.ipynb`](../notebooks/gpt.ipynb)

Место в линейке: **GPT-1** → [GPT-2](gpt2.md) → [LLaMA](llama.md) → [Mistral](mistral.md) → [Mixtral](mixtral.md) · [Gemma](gemma.md)

## Обзор

GPT-1 (Radford et al., [*"Improving Language Understanding by Generative Pre-Training"*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), OpenAI 2018) — первая архитектура, показавшая, что decoder-only трансформер, обученный на задаче предсказания следующего токена, переносится на широкий круг downstream-задач почти без изменения архитектуры. В этом репозитории воспроизведена "классическая" версия: обучаемые абсолютные позиционные эмбеддинги, стандартный multi-head attention и **post-LN** блок декодера (нормализация после residual-сложения — так, как было в оригинальной статье, до того как GPT-2 перешёл на pre-LN).

## Архитектура блока декодера

```mermaid
%%{init: {"flowchart": {"rankSpacing": 28, "nodeSpacing": 28}}}%%
flowchart TB
    Ids(["token ids"]):::io --> TokEmb["Token Embedding"]:::blue
    Ids --> PosEmb["Position Embedding<br/>(обучаемые)"]:::purple
    TokEmb --> Sum(("+")):::add
    PosEmb --> Sum
    Sum --> Drop["Dropout"]:::gray
    subgraph Dec["GptDecoder × num_layers · post-LN"]
        direction TB
        X(["x"]):::io --> Attn["Masked Multi-Head Attention"]:::blue
        Attn --> A1(("+")):::add
        X -. residual .-> A1
        A1 --> N1["LayerNorm"]:::gray
        N1 --> FFN["Feed Forward<br/>Linear → GELU → Linear"]:::purple
        FFN --> A2(("+")):::add
        N1 -. residual .-> A2
        A2 --> N2["LayerNorm"]:::gray
    end
    Drop --> Dec
    Dec --> Lin
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

Обратите внимание: `LayerNorm` стоит **после** сложения с residual-связью (`x + Attention(x)`, затем норма) — это ключевое отличие от GPT-2 и всех более поздних архитектур в этом репозитории, которые используют pre-LN.

## Устройство компонентов

### Multi-Head Attention

`h = num_heads` голов считаются параллельно; в коде это не отдельные модули, а одна проекция `Linear(emb_size, h · head_size)` для каждого из Q, K, V с последующим `reshape` на головы.

```mermaid
%%{init: {"flowchart": {"rankSpacing": 28, "nodeSpacing": 28}}}%%
flowchart TB
    X(["x · [batch, seq_len, emb_size]"]):::io
    X --> H1["Head 1"]:::blue
    X --> H2["Head 2"]:::blue
    X --> Hd["⋯"]:::io
    X --> Hh["Head h"]:::blue
    H1 --> Cat["Concat<br/>[batch, seq_len, h · head_size]"]:::gray
    H2 --> Cat
    Hh --> Cat
    Cat --> WO["Linear W_O → emb_size"]:::gray --> Drop["Dropout"]:::gray --> Out(["out"]):::io

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

### Одна голова: scaled dot-product attention с causal-маской

Маска запрещает позиции `i` смотреть на будущие позиции `j > i`: после `softmax` их веса становятся нулевыми.

```mermaid
%%{init: {"flowchart": {"rankSpacing": 28, "nodeSpacing": 28}}}%%
flowchart TB
    X(["x"]):::io --> Wq["W_q"]:::gray --> Q["Q"]:::blue
    X --> Wk["W_k"]:::gray --> K["K"]:::blue
    X --> Wv["W_v"]:::gray --> V["V"]:::blue
    Q --> QK["Q · Kᵀ"]:::gray
    K --> QK
    QK --> Scale["÷ √head_size"]:::gray
    Scale --> Mask["causal mask<br/>позиции j > i → −∞"]:::gold
    Mask --> SM["softmax по строкам"]:::purple
    SM --> AV["weights · V"]:::gray
    V --> AV
    AV --> O(["выход головы · [batch, seq_len, head_size]"]):::io

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

### Feed Forward

```mermaid
flowchart LR
    X(["x"]):::io --> L1["Linear<br/>emb_size → 4·emb_size"]:::gray --> Act["GELU"]:::purple --> L2["Linear<br/>4·emb_size → emb_size"]:::gray --> Drop["Dropout"]:::gray --> Out(["out"]):::io

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

## Компоненты

| Компонент | Класс | Файл |
|---|---|---|
| Токен-эмбеддинги | `TokenEmbeddings` | [`core/token_embeddings.py`](../llm/src/llm/core/token_embeddings.py) |
| Позиционные эмбеддинги | `PositionalEmbeddings` (обучаемые, абсолютные) | [`core/positional_embeddings.py`](../llm/src/llm/core/positional_embeddings.py) |
| Attention | `MultiHeadAttention` (стандартный causal MHA, без RoPE/GQA) | [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py) |
| FFN | `FeedForward` (2-слойный MLP, tanh-аппроксимация GELU — `activation="gelu_tanh"`, как в оригинальном коде OpenAI; меняется ключом `activation` в конфиге) | [`core/feed_forward.py`](../llm/src/llm/core/feed_forward.py) |
| Блок декодера | `GptDecoder` (**post-LN**) | [`core/gpt_decoder.py`](../llm/src/llm/core/gpt_decoder.py) |
| Модель целиком | `GPT` | [`models/gpt/gpt.py`](../llm/src/llm/models/gpt/gpt.py) |

`GptDecoder.forward`:
```
attn_out       = Attention(x)
out            = Norm1(attn_out + x)
ffn_out        = FFN(out)
result         = Norm2(ffn_out + out)
```

Важная деталь: после последнего блока декодера **нет** финальной нормализации — `GPT.forward` идёт напрямую из стека декодеров в `Linear`-проекцию на словарь. (GPT-2 в этом смысле отличается — см. [gpt2.md](gpt2.md).)

## Конфигурация

Пример из [`experiments/llm_only/configs/gpt_train.json`](../experiments/llm_only/configs/gpt_train.json):

| Параметр | Значение в примере | Смысл |
|---|---|---|
| `vocab_size` | (из токенизатора) | размер словаря |
| `embed_dim` | 256 | размерность эмбеддингов и скрытого состояния |
| `num_heads` | 4 | число attention-голов (`head_size = embed_dim / num_heads`) |
| `num_layers` | 4 | число блоков `GptDecoder` в стеке |
| `max_position_embeddings` | 128 | максимальная длина последовательности (размер буфера позиционных эмбеддингов и causal-маски) |
| `dropout` | 0.1 | dropout в attention и FFN |
| `activation` | (нет в примере) | необязательный: активация FFN — `"gelu_tanh"` (по умолчанию, tanh-аппроксимация GELU, как в оригинальном коде OpenAI), `"gelu"` (точный GELU через erf) или `"relu"` |

## Генерация

`GPT.generate(x, max_new_tokens, do_sample, temperature=1.0, top_k=None, top_p=None, use_cache=True, attention_mask=None, **kwargs)` — унифицированная сигнатура, общая для всех архитектур в этом репозитории: greedy (`do_sample=False`), sampling с температурой, top-k, top-p (nucleus), с опциональным KV-кэшем.

При генерации с KV-кэшем позиция новых токенов для позиционных эмбеддингов берётся из длины кэша (`cache_start_pos` в [`core/generation.py`](../llm/src/llm/core/generation.py)). Когда последовательность становится длиннее `max_position_embeddings`, `generate` берёт последние `max_position_embeddings` токенов и пересчитывает их без кэша: при сдвиге окна абсолютные позиции всех токенов меняются, и закэшированные K/V больше не годятся. `attention_mask` в `generate` допускается только из единиц — см. [Маски](README.md#attention_mask-и-паддинг).

## Что изменилось в GPT-2

- normalization: **post-LN → pre-LN**;
- появляется финальная нормализация перед выходной проекцией;
- FFN и attention переиспользуют ту же математику (GELU, стандартный MHA), но собраны в отдельный класс `Gpt2Decoder` вместо параметризуемого `GptDecoder`.

Подробности — в [gpt2.md](gpt2.md).

## Литература

Основная статья:

- Radford, Narasimhan, Salimans, Sutskever. *Improving Language Understanding by Generative Pre-Training*. OpenAI, 2018. [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (на arXiv не публиковалась)

Компоненты:

- Vaswani et al. *Attention Is All You Need*. 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Liu et al. *Generating Wikipedia by Summarizing Long Sequences*. 2018. [arXiv:1801.10198](https://arxiv.org/abs/1801.10198) — decoder-only трансформер, на который опирается GPT-1
- Hendrycks, Gimpel. *Gaussian Error Linear Units (GELUs)*. 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Ba, Kiros, Hinton. *Layer Normalization*. 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
