# Загрузка весов HuggingFace

[← Сохранение и загрузка](checkpoints.md) · [Оглавление](README.md) · [Интеграция с HuggingFace →](hf-integration.md)

Во все шесть моделей загружаются веса соответствующих моделей `transformers`. Для этого нужны:

1. **Конфиг, повторяющий оригинал.** По умолчанию модели библиотеки сохраняют прежнюю структуру (bias во всех `Linear`, скрытый размер FFN `4 · embed_dim`, отдельная выходная проекция). Ключи `bias`, `intermediate_size`, `tie_word_embeddings`, `rms_norm_eps`, `rope_theta` и другие делают модель такой же, как в HF.
2. **`convert_hf_state_dict`** из пакета модели: переименовывает ключи, а для моделей с RoPE ещё и переставляет строки `q_proj`/`k_proj` (RoPE здесь поворачивает соседние пары координат, как в коде Meta, а HF — половины вектора).

Нужен пакет `transformers` (ставится вместе с `hf-proxy`: `uv sync`). Все примеры проверены сверкой: логиты совпадают с HF до ~1e-5–1e-4, greedy-генерация с KV-кэшем — токен в токен. Для генерации текста нужен **токенизатор этой модели** из `transformers` (`AutoTokenizer.from_pretrained(...)`): собственный `BPETokenizer` даёт другие id.

## GPT-1 и GPT-2

```python
from transformers import GPT2LMHeadModel, OpenAIGPTLMHeadModel
from llm.models.gpt import GPT, GPT2, convert_hf_state_dict

hf = OpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")
gpt = GPT({"vocab_size": 40478, "embed_dim": 768, "num_heads": 12, "num_layers": 12,
           "max_position_embeddings": 512, "dropout": 0.0, "tie_word_embeddings": True})
gpt.load_state_dict(convert_hf_state_dict(hf.state_dict()))

hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
gpt2 = GPT2({"vocab_size": 50257, "embed_dim": 768, "num_heads": 12, "num_layers": 12,
             "max_position_embeddings": 1024, "dropout": 0.0, "tie_word_embeddings": True})
gpt2.load_state_dict(convert_hf_state_dict(hf.state_dict()))
```

Активация по умолчанию (`"gelu_tanh"`) совпадает с оригиналом у обеих моделей. Подробности — [GPT-1](../textbook/gpt.md#weight-tying-и-веса-openai), [GPT-2](../textbook/gpt2.md).

## LLaMA

```python
from transformers import LlamaForCausalLM
from llm.models.llama import Llama, convert_hf_state_dict

hf = LlamaForCausalLM.from_pretrained("nickypro/tinyllama-15M")
c = hf.config
model = Llama({"vocab_size": c.vocab_size, "embed_dim": c.hidden_size, "num_heads": c.num_attention_heads,
               "num_layers": c.num_hidden_layers, "max_position_embeddings": c.max_position_embeddings,
               "dropout": 0.0, "rms_norm_eps": c.rms_norm_eps, "rope_theta": c.rope_theta,
               "intermediate_size": c.intermediate_size, "bias": False})
model.load_state_dict(convert_hf_state_dict(hf.state_dict(), num_heads=c.num_attention_heads))
```

Подходят модели с обычным MHA (`num_key_value_heads == num_attention_heads`) и без `rope_scaling`. Подробности — [LLaMA](../textbook/llama.md#загрузка-весов-huggingface).

## Mistral и Mixtral

```python
from transformers import MistralForCausalLM
from llm.models.mistral import Mistral, convert_hf_state_dict

hf = MistralForCausalLM.from_pretrained(...)
c = hf.config
config = {"vocab_size": c.vocab_size, "embed_dim": c.hidden_size, "num_q_heads": c.num_attention_heads,
          "num_kv_heads": c.num_key_value_heads, "head_size": c.head_dim or c.hidden_size // c.num_attention_heads,
          "num_layers": c.num_hidden_layers, "max_position_embeddings": c.max_position_embeddings,
          "dropout": 0.0, "rms_norm_eps": c.rms_norm_eps, "rope_theta": c.rope_theta,
          "intermediate_size": c.intermediate_size, "bias": False}
if c.sliding_window is not None:
    config["window_size"] = c.sliding_window - 1
model = Mistral(config)
model.load_state_dict(convert_hf_state_dict(hf.state_dict(), num_heads=c.num_attention_heads,
                                            num_kv_heads=c.num_key_value_heads))
```

- **`window_size = sliding_window − 1`**: окно здесь на одну позицию шире, чем в HF (`window_size + 1` позиций, как в тексте статьи). Почему — [Ширина окна: W + 1](../textbook/mistral.md#ширина-окна-w--1).
- **Mixtral** — тот же код с классом `Mixtral` из `llm.models.mixtral` и двумя ключами: `"num_experts": c.num_local_experts`, `"top_k_experts": c.num_experts_per_tok`. Подробности — [Mixtral](../textbook/mixtral.md#загрузка-весов-huggingface).

## Gemma

```python
from transformers import GemmaForCausalLM
from llm.models.gemma import Gemma, convert_hf_state_dict

hf = GemmaForCausalLM.from_pretrained("google/gemma-2b")
c = hf.config
model = Gemma({"vocab_size": c.vocab_size, "embed_dim": c.hidden_size, "num_q_heads": c.num_attention_heads,
               "num_kv_heads": c.num_key_value_heads, "head_size": c.head_dim, "num_layers": c.num_hidden_layers,
               "max_position_embeddings": c.max_position_embeddings, "dropout": 0.0,
               "rms_norm_eps": c.rms_norm_eps, "rope_theta": c.rope_theta, "intermediate_size": c.intermediate_size,
               "bias": False, "tie_word_embeddings": True, "scale_embeddings": True})
model.load_state_dict(convert_hf_state_dict(hf.state_dict(), num_heads=c.num_attention_heads,
                                            num_kv_heads=c.num_key_value_heads))
```

`google/gemma-2b` закрыт лицензией: доступ — после принятия условий на HuggingFace и входа через `huggingface-cli login`. `convert_hf_state_dict` Gemma прибавляет 1 к весам RMSNorm (`GemmaRMSNorm` умножает на `1 + w`). Подробности — [Gemma](../textbook/gemma.md#загрузка-весов-huggingface).

## После загрузки

```python
model.eval()
model.save("gpt2.pt")    # дальше — Model.load без transformers
```

- Ошибки `load_state_dict` вида «size mismatch» или «missing keys» почти всегда значат, что конфиг не совпадает с оригиналом: проверьте `bias`, `intermediate_size`, `tie_word_embeddings`, `head_size`.
- `convert_hf_state_dict` бросает `KeyError` на незнакомый ключ чекпоинта — значит, у модели есть слои, которых в библиотеке нет (например, другая архитектура под тем же классом HF).
- В bfloat16 и float16 возможны расхождения с HF в последних битах.
