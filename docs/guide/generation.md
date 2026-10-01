# Генерация
<!-- description: Метод generate у всех шести моделей: greedy и сэмплирование, батч промптов, остановка по EOS. -->

[← Обучение](training.md) · [Оглавление](README.md) · [Сохранение и загрузка →](checkpoints.md)

У всех шести моделей один метод `generate` из `BaseModel`:

```python
generate(x, max_new_tokens, do_sample, temperature=1.0, top_k=None, top_p=None, use_cache=True,
         attention_mask=None, eos_token_id=None, pad_token_id=None)
```

Он возвращает `LongTensor [batch, prompt_len + n]` — промпт вместе с `n` новыми токенами (`n = max_new_tokens` или меньше при остановке по EOS). Градиенты не считаются; неизвестный именованный аргумент — `TypeError`.

## Пример

```python
import torch

model.eval()                                            # выключить dropout
prompt = torch.tensor([tokenizer.encode("Нейронные сети")])
out = model.generate(prompt, max_new_tokens=20, do_sample=True, temperature=0.8, top_p=0.9)
print(tokenizer.decode(out[0].tolist()))
```

## Способы выбора токена

| Параметры | Что происходит |
|---|---|
| `do_sample=False` | greedy: всегда самый вероятный токен; `temperature`, `top_k`, `top_p` не используются |
| `do_sample=True, temperature=τ` | сэмплирование из softmax(логиты / τ); τ < 1 — увереннее, τ > 1 — разнообразнее; τ ≤ 0 — `ValueError` |
| `+ top_k=k` | только из `k` самых вероятных токенов |
| `+ top_p=p` | nucleus: из минимального набора токенов с суммарной вероятностью ≥ `p` (как в HF); `p` из (0, 1] |

`top_k` и `top_p` вместе — `ValueError`. Для воспроизводимого сэмплирования задайте `torch.manual_seed(...)` перед вызовом. Как устроен каждый способ — в главе [Генерация текста](../textbook/generation.md).

## Остановка по EOS

```python
out = model.generate(prompt, max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id)
```

Строка, сгенерировавшая `eos_token_id`, считается законченной, и дальше в неё пишется `pad_token_id` (по умолчанию тот же `eos_token_id`); генерация останавливается, когда закончены все строки. Без `eos_token_id` генерируется ровно `max_new_tokens` токенов.

## Батч промптов разной длины

Промпты дополняются **слева**, чтобы новые токены дописывались сразу после текста каждой строки, и передаётся `attention_mask`:

```python
prompts = [tokenizer.encode("Нейронные сети"), tokenizer.encode("Трансформеры")]
width = max(len(p) for p in prompts)
pad = tokenizer.pad_token_id
ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts])
mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
out = model.generate(ids, max_new_tokens=20, do_sample=False, attention_mask=mask)
```

Каждая строка даёт то же, что её промпт, сгенерированный отдельно. Правый паддинг в `generate` — `ValueError`: генерация продолжилась бы с pad-токена.

## KV-кэш и длинные тексты

- `use_cache=True` (по умолчанию) — на каждом шаге в модель подаётся только новый токен, а K и V прошлых токенов берутся из кэша; результат тот же, генерация быстрее.
- Когда последовательность становится длиннее `max_position_embeddings`, `generate` продолжает по последним `max_position_embeddings` токенам и пересчитывает их без кэша. Модель при этом не видит начало текста.
- У Mistral и Mixtral со скользящим окном кэш обрезается до окна, и память на кэш не растёт с длиной текста.
