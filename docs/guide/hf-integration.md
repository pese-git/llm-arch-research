# Интеграция с HuggingFace
<!-- description: Пакет hf-proxy оборачивает модели библиотеки в интерфейсы transformers: обучение через transformers.Trainer и HF-формат (только GPT). -->

[← Загрузка весов HuggingFace](hf-weights.md) · [Оглавление](README.md) · [Ограничения →](limitations.md)

Пакет `hf-proxy` оборачивает модель и токенизатор библиотеки в интерфейсы `transformers`: `PreTrainedModel`, `PretrainedConfig`, HF-подобный токенизатор. С ним собственная модель обучается через `transformers.Trainer`, сохраняется в HF-формате и работает с коллаторами `transformers`.

> **Поддерживается только `GPT`** (`llm.models.gpt.GPT`). GPT-2, LLaMA, Mistral, Mixtral и Gemma через hf-proxy не работают. Загрузить веса HF в любую из шести моделей можно и без hf-proxy — см. [Загрузку весов HuggingFace](hf-weights.md). API экспериментальный.

## Обернуть модель и токенизатор

```python
import torch
from llm.models.gpt import GPT
from llm.tokenizers import BPETokenizer
from hf_proxy import HFAdapter, HFTokenizerAdapter

tokenizer = BPETokenizer.load("checkpoints/bpe_tokenizer.json")
model = GPT({"vocab_size": tokenizer.get_vocab_size(), "embed_dim": 256, "num_heads": 4,
             "num_layers": 4, "max_position_embeddings": 128, "dropout": 0.1})

hf_model = HFAdapter.from_llm_model(model)        # HFGPTAdapter — наследник PreTrainedModel
hf_tokenizer = HFTokenizerAdapter(tokenizer)

input_ids = torch.tensor([tokenizer.encode("Нейронные сети")])
out = hf_model(input_ids=input_ids, labels=input_ids)   # loss считается при переданных labels
print(out.loss, out.logits.shape)
generated = hf_model.generate(input_ids=input_ids, max_new_tokens=30, do_sample=True, temperature=0.8)
```

## Обучение через transformers.Trainer

`HFTokenizerAdapter.pad` совместим с коллаторами `transformers`: `input_ids` дополняются pad-токеном, `attention_mask` — нулями, `labels` — значением `-100`.

```python
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)
args = TrainingArguments(output_dir="checkpoints/hf-trained", num_train_epochs=3,
                         per_device_train_batch_size=2, learning_rate=3e-4, warmup_ratio=0.1)
trainer = Trainer(model=hf_model, args=args, train_dataset=dataset, data_collator=collator)
trainer.train()
```

`dataset` — любой набор словарей с `input_ids` (например, `datasets.Dataset` после токенизации). Полный сценарий — `experiments/hf_integration/train_with_hf_trainer.py`.

## Сохранение и загрузка в HF-формате

```python
HFAdapter.save_pretrained(hf_model, "checkpoints/my-gpt", tokenizer=hf_tokenizer)   # config.json, pytorch_model.bin, токенизатор

loaded = HFAdapter.from_pretrained("checkpoints/my-gpt/pytorch_model.bin")
loaded_tokenizer = HFTokenizerAdapter.from_pretrained("checkpoints/my-gpt")
```

`from_pretrained` читает `config.json` рядом с весами. Без него размеры восстанавливаются по весам, а число голов внимания берётся по умолчанию (12) с предупреждением: по весам его не определить; тогда передайте `hf_config` явно. Токенизатор сохраняется вместе со слияниями BPE и после загрузки кодирует так же.

## Что не поддерживается

- KV-кэш HF (`past_key_values`) игнорируется.
- `HFGPTAdapter.generate` передаёт управление `GPT.generate`: учитываются `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `use_cache`, `eos_token_id`, `pad_token_id`; другой именованный аргумент — `TypeError`. `generation_config`, `logits_processor` и `stopping_criteria` не применяются.
- Значения по умолчанию в `HFAdapterConfig` (`pad/bos/eos_token_id = 50256`) рассчитаны на словарь GPT-2 и не соответствуют собственному BPE-токенизатору — задавайте их явно.

Справочник по объектам пакета — [hf-proxy/README.md](../../hf-proxy/README.md). Готовые скрипты — [experiments/hf_integration](../../experiments/README.md#-hf_integration-через-hf-proxy).
