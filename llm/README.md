# llm — библиотека архитектур LLM

Модульная учебная библиотека на PyTorch: строительные блоки трансформера и шесть собранных из них моделей — **GPT, GPT-2, LLaMA, Mistral, Mixtral, Gemma**. Зависит только от `torch` и `numpy`.

Разбор каждой архитектуры — в [../docs/](../docs/README.md).

## 🏗️ Структура

```
src/llm/
├── core/                         # строительные блоки
│   ├── base_model.py             # BaseModel — абстрактный базовый класс
│   ├── token_embeddings.py       # TokenEmbeddings
│   ├── positional_embeddings.py  # PositionalEmbeddings — обучаемые абсолютные позиции (GPT, GPT-2)
│   ├── rope.py                   # RoPE — Rotary Positional Embeddings
│   ├── multi_head_attention.py   # MultiHeadAttention (+ опциональный RoPE, KV-кэш)
│   ├── multi_query_attention.py  # MultiQueryAttention — одна общая K/V-голова (Gemma)
│   ├── group_query_attention.py  # GroupedQueryAttention + sliding window (Mistral, Mixtral)
│   ├── feed_forward.py           # FeedForward с GELU
│   ├── swi_glu.py                # SwiGLU
│   ├── geglu.py                  # GeGLU
│   ├── gelu.py, silu.py          # активации
│   ├── rms_norm.py               # RMSNorm
│   ├── moe.py                    # MoE — top-k роутинг по SwiGLU-экспертам
│   ├── cached_decoder.py         # CachedDecoder — параметризуемый pre-LN блок (LLaMA)
│   ├── gpt_decoder.py            # GptDecoder (post-LN)
│   ├── gpt2_decoder.py           # Gpt2Decoder (pre-LN)
│   ├── mistral_decoder.py        # MistralDecoder
│   ├── mixtral_decoder.py        # MixtralDecoder
│   └── gemma_decoder.py          # GemmaDecoder
├── models/
│   ├── gpt/                      # GPT, GPT2
│   ├── llama/                    # Llama
│   ├── mistral/                  # Mistral
│   ├── mixtral/                  # Mixtral
│   └── gemma/                    # Gemma
├── tokenizers/                   # BaseTokenizer, BPETokenizer, SimpleBPETokenizer
├── datasets/                     # TextDataset, StreamingTextDataset, TextWithSpecialTokensDataset
├── training/                     # Trainer, get_optimizer, get_linear_schedule_with_warmup
└── evaluation/                   # заготовка, пока пустая
```

## 🏆 Архитектуры

| Модель | Класс | Attention | Позиции | Норма | FFN | Блок |
|---|---|---|---|---|---|---|
| GPT | `llm.models.gpt.GPT` | MHA | обучаемые | LayerNorm, post-LN | GELU | `GptDecoder` |
| GPT-2 | `llm.models.gpt.GPT2` | MHA | обучаемые | LayerNorm, pre-LN + финальная | GELU | `Gpt2Decoder` |
| LLaMA | `llm.models.llama.Llama` | MHA | RoPE | RMSNorm | SwiGLU | `CachedDecoder` |
| Mistral | `llm.models.mistral.Mistral` | GQA + sliding window | RoPE | RMSNorm | SwiGLU | `MistralDecoder` |
| Mixtral | `llm.models.mixtral.Mixtral` | GQA + sliding window | RoPE | RMSNorm | MoE (SwiGLU) | `MixtralDecoder` |
| Gemma | `llm.models.gemma.Gemma` | MQA | RoPE | RMSNorm | GeGLU | `GemmaDecoder` |

### Ключи конфига

Конфиг модели — обычный `dict`. Размер головы — ключ `head_size`, а если его нет, `embed_dim // <число голов>` (тогда `embed_dim` должен делиться на число голов). Неверный конфиг — неделимые размеры, `num_q_heads`, не делящееся на `num_kv_heads`, нечётный `head_size` в моделях с RoPE, `top_k_experts` вне `1 … num_experts` — даёт `ValueError` в конструкторе.

| Ключ | GPT, GPT-2, LLaMA | Mistral | Mixtral | Gemma |
|---|---|---|---|---|
| `vocab_size`, `embed_dim`, `num_layers`, `max_position_embeddings`, `dropout` | ✅ | ✅ | ✅ | ✅ |
| `num_heads` | ✅ | | | |
| `num_q_heads` | | ✅ | ✅ | ✅ |
| `num_kv_heads` | | ✅ | ✅ | |
| `head_size` (необязательный) | ✅ | ✅ | ✅ | ✅ |
| `rms_norm_eps` (необязательный, по умолчанию `1e-6`) | LLaMA | ✅ | ✅ | ✅ |
| `rope_theta` (необязательный, по умолчанию `10000`) | LLaMA | ✅ | ✅ | ✅ |
| `router_aux_loss_coef` (необязательный, по умолчанию `0`) | | | ✅ | |
| `tie_word_embeddings` (необязательный, по умолчанию `false`) | GPT, GPT-2 | | | |
| `window_size` | | ✅ | ✅ | |
| `num_experts`, `top_k_experts` | | | ✅ | |

## 🧩 Ключевые компоненты

### CachedDecoder (`core/cached_decoder.py`)
**Универсальный декодер** с поддержкой dependency injection и кэширования KV-памяти.

```python
CachedDecoder(
    feed_forward_layer=FeedForward(...),  # или SwiGLU
    norm_layer=nn.LayerNorm,              # или RMSNorm
    rope=RoPE(...),                       # опционально
    # ... другие параметры
)
```

### RoPE (`core/rope.py`)
**Rotary Positional Embeddings** - ротационные позиционные эмбеддинги.

**Математическая основа:**
```
θ_i = base^(-2i/d)
q'_m = q_m * cos(mθ_i) + rotate(q_m) * sin(mθ_i)
```

### RMSNorm (`core/rms_norm.py`)
**Root Mean Square Normalization** - упрощенная нормализация без среднего.

**Формула:**
```
RMSNorm(x) = (x / RMS(x)) * w
где RMS(x) = sqrt(mean(x²) + eps)
```

### SwiGLU (`core/swi_glu.py`)
**Swish-Gated Linear Unit** - современная активация с gating mechanism.

**Формула:**
```
SwiGLU(x) = Swish(xW_g + b_g) ⊙ (xW_u + b_u) * W_d + b_d
```

## 🚀 Примеры

### Создание модели и forward

```python
import torch
from llm.models.llama import Llama

model = Llama({
    "vocab_size": 32000,
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "max_position_embeddings": 1024,
    "dropout": 0.1,
})

input_ids = torch.randint(0, 32000, (2, 16))

# Все модели возвращают кортеж (logits, cache).
# logits: [batch, seq_len, vocab_size]; cache: список (K, V) по слоям или None
logits, cache = model(input_ids, use_cache=False)
```

### Генерация

У всех моделей одинаковая сигнатура:

```python
generate(x, max_new_tokens, do_sample, temperature=1.0, top_k=None, top_p=None, use_cache=True, attention_mask=None, eos_token_id=None, pad_token_id=None)
```

```python
# Greedy
out = model.generate(input_ids, max_new_tokens=50, do_sample=False)

# Sampling с температурой и top-p
out = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.8, top_p=0.9)
```

Когда последовательность становится длиннее `max_position_embeddings`, `generate` продолжает по последним `max_position_embeddings` токенам. `attention_mask` допускается только из единиц; `forward` принимает и правый паддинг (подробнее — [docs/README.md](../docs/README.md#маски)).

### Сохранение и загрузка

```python
model.save("checkpoints/mistral.pt")                     # класс, конфиг и веса в одном файле
model = Mistral.load("checkpoints/mistral.pt", device="cuda")  # конфиг передавать не нужно
```

`load` — метод класса: модель создаётся по сохранённому конфигу и возвращается в режиме `eval`. Файл читается с `weights_only=True`. Маски attention и таблицы RoPE в файл не попадают — они вычисляются из конфига.

### Токенизатор и обучение

```python
from llm.tokenizers import BPETokenizer
from llm.datasets.text_dataset import TextDataset
from llm.training.trainer import Trainer

texts = ["Первый текст для обучения.", "Второй текст для обучения."]

tokenizer = BPETokenizer()
tokenizer.train(texts=texts, vocab_size=300, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
tokenizer.save("bpe_tokenizer.json")

dataset = TextDataset(texts, tokenizer, block_size=64)
trainer = Trainer(model=model, train_dataset=dataset, lr=3e-4, batch_size=8, num_epochs=3, warmup_steps=100)
trainer.train()
```

`BPETokenizer` перед обучением и кодированием разбивает текст на слова, как GPT-2: пробел прикрепляется к началу следующего слова, пунктуация идёт отдельно (`pretokenize` в `bpe_tokenizer.py`). Слияния не выходят за границы слов, поэтому токен не длиннее слова. На маленьком корпусе обучение может остановиться раньше `vocab_size`, когда каждое слово уже стало одним токеном.

`Trainer` — минимальный цикл: AdamW, линейный warmup/decay, gradient clipping 1.0, устройство `cuda` или `cpu`. Сохранение чекпоинтов, AMP и gradient accumulation в нём не реализованы.

## ⚠️ Известные ограничения

- **`attention_mask`: только правый паддинг** — на левый паддинг и паддинг в `generate` бросается `NotImplementedError`.

## 🧪 Тестирование

```bash
cd llm
uv run pytest
```

Около 250 тестов, покрывающих все блоки `core/`, модели, токенизаторы, датасеты и обучение.

## 📚 Научные концепции

### Трансформерная архитектура
Основана на механизме **внимания**, позволяющем модели взвешивать важность разных частей входной последовательности.

**Формула внимания:**
```
Attention(Q, K, V) = softmax(Q·Kᵀ/√d_k)·V
```

### RoPE (Rotary Positional Embeddings)
Инновационный метод кодирования позиционной информации через **вращение векторов** в комплексном пространстве.

**Преимущества:**
- Относительное позиционное кодирование
- Лучшая экстраполяция на длинные последовательности
- Сохранение нормы векторов

### RMSNorm vs LayerNorm
**RMSNorm** устраняет вычитание среднего, что делает его более стабильным и эффективным при обучении больших моделей.

### SwiGLU vs GELU
**SwiGLU** с gating mechanism показывает лучшую производительность благодаря способности выборочно передавать информацию.

## 🔧 Добавление новой архитектуры

1. Соберите блок декодера из компонентов `core/` (или используйте `CachedDecoder`, передав `norm_layer` и `feed_forward_layer`).
2. Создайте класс модели, наследующий `BaseModel`, с `forward(x, use_cache=False, cache=None, attention_mask=None) -> (logits, cache)` и атрибутом `_max_seq_len`; `generate` наследуется из `BaseModel`.
3. Добавьте тесты в `tests/core/` и `tests/models/`.
4. Зарегистрируйте модель в `experiments/llm_only/run_llm_experiment.py` и добавьте конфиги в `experiments/llm_only/configs/`.

## 📄 Лицензия

MIT License
