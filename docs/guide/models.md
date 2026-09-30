# Модели и конфиги

[← Установка](installation.md) · [Оглавление](README.md) · [Токенизатор и данные →](data.md)

## Шесть моделей

| Класс | Импорт | Что отличает архитектуру | Глава пособия |
|---|---|---|---|
| `GPT` | `from llm.models.gpt import GPT` | обучаемые позиции, MHA, post-LN, GELU-FFN | [GPT-1](../textbook/gpt.md) |
| `GPT2` | `from llm.models.gpt import GPT2` | то же + pre-LN, финальный LayerNorm | [GPT-2](../textbook/gpt2.md) |
| `Llama` | `from llm.models.llama import Llama` | RoPE, RMSNorm, SwiGLU, MHA | [LLaMA](../textbook/llama.md) |
| `Mistral` | `from llm.models.mistral import Mistral` | + GQA и скользящее окно | [Mistral](../textbook/mistral.md) |
| `Mixtral` | `from llm.models.mixtral import Mixtral` | Mistral + Mixture-of-Experts | [Mixtral](../textbook/mixtral.md) |
| `Gemma` | `from llm.models.gemma import Gemma` | MQA (или GQA/MHA), GeGLU, масштаб эмбеддингов | [Gemma](../textbook/gemma.md) |

Все шесть — наследники `BaseModel` (`llm.core.base_model`) и `torch.nn.Module` с одинаковым интерфейсом: `forward`, `generate`, `save`, `load`, `auxiliary_loss`, свойство `max_seq_len`.

## Конфиг

Модель создаётся из обычного `dict`:

```python
from llm.models.mistral import Mistral

model = Mistral({
    "vocab_size": 1000,               # размер словаря — из токенизатора: tokenizer.get_vocab_size()
    "embed_dim": 256,                 # d — размерность модели
    "num_q_heads": 4,                 # головы запросов
    "num_kv_heads": 2,                # головы ключей и значений (GQA)
    "num_layers": 4,
    "max_position_embeddings": 128,   # максимальная длина контекста
    "window_size": 16,                # скользящее окно; без ключа окна нет
    "dropout": 0.1,
})
print(sum(p.numel() for p in model.parameters()))   # число параметров
```

### Ключи

✅ — ключ обязательный, «необяз.» — есть значение по умолчанию, пусто — ключ модели не нужен (лишние ключи игнорируются).

| Ключ | GPT, GPT-2 | LLaMA | Mistral | Mixtral | Gemma |
|---|---|---|---|---|---|
| `vocab_size`, `embed_dim`, `num_layers`, `max_position_embeddings`, `dropout` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `num_heads` | ✅ | ✅ | | | |
| `num_q_heads` | | | ✅ | ✅ | ✅ |
| `num_kv_heads` | | | ✅ | ✅ | необяз., `1` (MQA) |
| `num_experts`, `top_k_experts` | | | | ✅ | |
| `head_size` | необяз. | необяз. | необяз. | необяз. | необяз. |
| `window_size` | | | необяз., без окна | необяз., без окна | |
| `rms_norm_eps` | | `1e-6` | `1e-6` | `1e-6` | `1e-6` |
| `rope_theta` | | `10000` | `10000` | `10000` | `10000` |
| `intermediate_size` | | `4 · embed_dim` | `4 · embed_dim` | `4 · embed_dim` | `4 · embed_dim` |
| `bias` | | `true` | `true` | `true` | `true` |
| `tie_word_embeddings` | `false` | | | | `false` |
| `scale_embeddings` | | | | | `false` |
| `activation` | только GPT-1: `"gelu_tanh"` или `"gelu"` (точный) | | | | |
| `attention_dropout` | `0.0` | | | | |
| `router_aux_loss_coef` | | | | `0.0` | |
| `initializer_range` | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` |

Смысл ключей:

- **`head_size`** — размер одной головы; без ключа `embed_dim // <число голов>`. В моделях с RoPE должен быть чётным.
- **`intermediate_size`**, **`bias`** — по умолчанию сохранена прежняя структура (`4 · embed_dim`, bias во всех `Linear`), чтобы загружались старые чекпоинты. У оригинальных LLaMA, Mistral, Mixtral и Gemma bias нет и скрытый размер FFN другой — эти ключи нужны для [загрузки весов HF](hf-weights.md).
- **`tie_word_embeddings`** — общая матрица эмбеддингов и выходной проекции, как в оригинальных GPT и Gemma.
- **`scale_embeddings`** — умножение эмбеддингов на √`embed_dim`, как в Gemma.
- **`router_aux_loss_coef`** — коэффициент load-balancing loss роутера Mixtral; `0` — выключен (в HF — `0.001`). Подробнее — в [Обучении](training.md#mixtral-load-balancing-loss).
- **`initializer_range`** — стандартное отклонение начальных весов `Linear` и `Embedding`, как в HF. Важно только для обучения с нуля.

**Проверки.** Неверный конфиг даёт `ValueError` уже в конструкторе: `embed_dim` не делится на число голов (без явного `head_size`), `num_q_heads` не делится на `num_kv_heads`, нечётный `head_size` в моделях с RoPE, `top_k_experts` вне `1 … num_experts`, `intermediate_size ≤ 0`, `rms_norm_eps ≤ 0`.

Параметры оригинальных моделей (размеры 7B, 8x7B и т. п.) и что меняет каждый ключ — в главах пособия, раздел «Конфигурация» в главе нужной архитектуры.

## Прямой проход

```python
import torch

x = torch.randint(0, 1000, (2, 16))          # [batch, seq_len]
logits, cache = model(x)                      # logits: [2, 16, vocab_size]; cache = None
logits, cache = model(x, use_cache=True)      # cache — список по слоям для продолжения
next_logits, cache = model(x[:, -1:], use_cache=True, cache=cache)
```

- `forward` всегда возвращает кортеж `(logits, cache)`; кэш — только при `use_cache=True`.
- **Аргументы передавайте по имени.** Порядок позиционных аргументов разный: у `GPT` — `(x, attention_mask, use_cache, cache)`, у остальных — `(x, use_cache, cache, attention_mask)`.
- **`attention_mask`** `[batch, seq_len]` (1 — токен, 0 — паддинг): паддинг допускается в любом месте строки. С кэшем маска должна покрывать и кэш: `[batch, cache_len + seq_len]`. Как это устроено — в главе [Маски](../textbook/masks.md#attention_mask-и-паддинг).
- Длина с учётом кэша не больше `max_position_embeddings`, иначе `ValueError`. `generate` сам обрезает контекст — см. [Генерацию](generation.md).
- Формат кэша: у GPT, GPT-2 и LLaMA — пара `(K, V)` на слой, у Mistral, Mixtral и Gemma — тройка `(K, V, next_pos)` (кэш со скользящим окном обрезается, и позиция хранится отдельно). Разбирать его вручную обычно не нужно: передавайте кэш из одного вызова в следующий как есть.

Логиты — не вероятности: для вероятностей нужен `softmax(logits, dim=-1)`, а `cross_entropy` принимает логиты напрямую.
