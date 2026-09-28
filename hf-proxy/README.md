# hf-proxy — адаптер llm ↔ HuggingFace

Экспериментальный пакет, который оборачивает модели и токенизаторы библиотеки [`llm`](../llm/README.md) в интерфейсы HuggingFace Transformers: `PreTrainedModel`, `PretrainedConfig`, HF-подобный токенизатор. Это позволяет использовать собственные модели с `transformers.Trainer` и сохранять их в HF-формате.

> ⚠️ **Поддерживается только модель `GPT`** (`llm.models.gpt.GPT`). GPT-2, LLaMA, Mistral, Mixtral и Gemma через hf-proxy не работают.
> Загрузить предобученные веса с HuggingFace Hub (например, `mistralai/Mistral-7B-v0.1`) в модели `llm` нельзя.
> API экспериментальный и может меняться; совместимость с будущими версиями Transformers не гарантируется.

Зависимости: `torch>=2.3.0`, `transformers>=4.44.0`, `datasets>=2.20.0`.

## 🧩 Состав

| Объект | Файл | Назначение |
|---|---|---|
| `HFAdapter` | `hf_adapter.py` | фабрика: `from_llm_model`, `from_pretrained`, `save_pretrained` |
| `HFGPTAdapter` | `hf_adapter.py` | наследник `PreTrainedModel`, оборачивает `GPT`; `forward` возвращает `CausalLMOutputWithCrossAttentions` с `loss` при переданных `labels` |
| `HFAdapterConfig` | `hf_config.py` | dataclass-конфиг; `from_llm_config()` переводит ключи `llm` в имена HF (`embed_dim` → `hidden_size` и т.д.) |
| `HFPretrainedConfig` | `hf_config.py` | наследник `PretrainedConfig` (`model_type = "gpt"`) |
| `HFTokenizerAdapter` | `hf_tokenizer.py` | HF-подобный интерфейс над `BaseTokenizer`: `encode`, `decode`, `pad`, `tokenize`, `save_pretrained`, `from_pretrained` |
| `create_hf_tokenizer`, `convert_to_hf_format` | `hf_tokenizer.py` | обёртка токенизатора и его сохранение в HF-формате |
| `HFUtils`, `TokenizerWrapper`, `create_hf_pipeline` | `hf_utils.py` | конвертация, `push_to_hub` / `load_from_hub`, сравнение с HF-моделью, создание `pipeline` |

## 🚀 Использование

### Обернуть модель и токенизатор

```python
import torch
from llm.models.gpt import GPT
from llm.tokenizers import BPETokenizer
from hf_proxy import HFAdapter, HFTokenizerAdapter

tokenizer = BPETokenizer.load("checkpoints/bpe_tokenizer.json")
model = GPT({
    "vocab_size": tokenizer.get_vocab_size(),
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 4,
    "max_position_embeddings": 128,
    "dropout": 0.1,
})

hf_model = HFAdapter.from_llm_model(model)        # HFGPTAdapter
hf_tokenizer = HFTokenizerAdapter(tokenizer)

input_ids = torch.tensor([tokenizer.encode("Привет")])
out = hf_model(input_ids=input_ids, labels=input_ids)
print(out.loss, out.logits.shape)

generated = hf_model.generate(input_ids=input_ids, max_new_tokens=30, do_sample=True, temperature=0.8)
```

### Сохранение и загрузка

```python
HFAdapter.save_pretrained(hf_model, "checkpoints/my-gpt")     # config.json + pytorch_model.bin
hf_tokenizer.save_pretrained("checkpoints/my-gpt-tokenizer")

from hf_proxy import HFAdapterConfig
config = HFAdapterConfig.from_llm_config(model.config)
loaded = HFAdapter.from_pretrained("checkpoints/my-gpt/pytorch_model.bin", hf_config=config)
loaded_tokenizer = HFTokenizerAdapter.from_pretrained("checkpoints/my-gpt-tokenizer")
```

Передавайте `hf_config` в `HFAdapter.from_pretrained` явно: без него из весов восстанавливаются только `vocab_size` и `embed_dim`, а число слоёв, голов и длина контекста берутся по умолчанию (12 / 12 / 1024); если они не совпадают с сохранённой моделью, `load_state_dict` падает с `RuntimeError`.

Готовые сценарии — в [experiments/hf_integration/](../experiments/README.md#-hf_integration-через-hf-proxy).

## ⚠️ Известные ограничения

- **Только `GPT`.** `HFAdapter` всегда создаёт `llm.models.gpt.GPT`.
- **`attention_mask` и `past_key_values` игнорируются** в `forward`; KV-кэш HF не поддерживается.
- **`HFGPTAdapter.generate`** передаёт управление `GPT.generate`: учитываются `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, а `generation_config`, `logits_processor`, `stopping_criteria` и остановка по `eos_token_id` не применяются.
- **`GPT.generate` с KV-кэшем (включён по умолчанию) генерирует неверно** — позиции новых токенов не сдвигаются, см. [docs/gpt.md](../docs/gpt.md#генерация). Передавайте `use_cache=False`.
- **`HFAdapter.save_pretrained(model, dir, tokenizer=...)` не сохраняет токенизатор** — сохраняйте его отдельно через `hf_tokenizer.save_pretrained(...)`.
- Значения по умолчанию в `HFAdapterConfig` (`pad/bos/eos_token_id = 50256`, `architectures = ["GPT2LMHeadModel"]`) рассчитаны на словарь GPT-2 и не соответствуют собственному BPE-токенизатору.
