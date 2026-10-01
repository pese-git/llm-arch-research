# Обучение
<!-- description: Обучение моделей библиотеки llm: Trainer, оптимизатор, расписание learning rate и скрипт экспериментов. -->

[← Токенизатор и данные](data.md) · [Оглавление](README.md) · [Генерация →](generation.md)

## Минимальный пример

```python
from llm.datasets.text_dataset import TextDataset
from llm.models.llama import Llama
from llm.tokenizers import BPETokenizer
from llm.training.trainer import Trainer

texts = ["Нейронные сети учатся на данных.", "Трансформеры обрабатывают последовательности."] * 8

tokenizer = BPETokenizer()
tokenizer.train(texts=texts, vocab_size=200, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

model = Llama({"vocab_size": tokenizer.get_vocab_size(), "embed_dim": 64, "num_heads": 4,
               "num_layers": 2, "max_position_embeddings": 32, "dropout": 0.1})
train = TextDataset(texts, tokenizer, block_size=32)

trainer = Trainer(model, train, val_dataset=train, lr=1e-3, batch_size=4, num_epochs=3, warmup_ratio=0.1)
trainer.train()                  # печатает средний loss эпохи и валидационный loss
print(trainer.loss_history)      # средний train loss по эпохам
model.save("llama.pt")           # класс, конфиг и веса в одном файле; папку save не создаёт
```

## Trainer

```python
Trainer(model, train_dataset, val_dataset=None, lr=3e-4, batch_size=8, num_epochs=3,
        warmup_steps=None, warmup_ratio=None)
```

| Параметр | Что задаёт |
|---|---|
| `lr` | пиковый learning rate |
| `batch_size`, `num_epochs` | размер батча и число эпох; обучающие данные перемешиваются каждую эпоху |
| `warmup_steps` / `warmup_ratio` | длина линейного warmup: числом шагов или долей от всех шагов, `ceil(N_steps · warmup_ratio)`, как в HF. Оба сразу — `ValueError`; ни одного — 100 шагов |
| `val_dataset` | если задан, после каждой эпохи вызывается `evaluate()` |

Что делает `train()`:

- **Оптимизатор** — AdamW, `weight_decay=0.01` только на матрицах (веса `Linear` и эмбеддинги); bias и веса нормализаций не затухают. Бета — значения PyTorch по умолчанию, `(0.9, 0.999)`.
- **Расписание** — линейный warmup от 0 до `lr`, затем линейный спад до 0 к концу обучения. Если warmup не короче всего обучения, `train()` предупреждает: learning rate не дойдёт до заданного.
- **Loss** — cross-entropy следующего токена: логиты сдвигаются на одну позицию относительно меток, позиции с меткой `-100` не учитываются. Батч без единой цели даёт loss 0, а не NaN.
- **Gradient clipping** — общая норма градиента обрезается до 1.0.
- **`attention_mask`** из батча передаётся в модель, если она есть.
- **Устройство** — `cuda`, если доступна, иначе `cpu`; модель переносится туда в конструкторе.

`evaluate()` переводит модель в режим eval, считает средний loss по валидационному набору без градиентов и возвращает его. После `train()` модель остаётся в режиме eval, если была валидация, и в режиме train — если не было; перед генерацией вызывайте `model.eval()`.

Почему именно так — AdamW, warmup, clipping, инициализация, — в главе [Обучение](../textbook/training.md).

### Свой оптимизатор

Оптимизатор создаётся в конструкторе, а планировщик — в `train()` по `trainer.optimizer`, поэтому оптимизатор можно подменить до `train()`:

```python
import torch
from llm.training.optimizer import weight_decay_param_groups

trainer = Trainer(model, train, lr=3e-4, batch_size=4, num_epochs=3, warmup_ratio=0.1)
trainer.optimizer = torch.optim.AdamW(weight_decay_param_groups(model, 0.1), lr=3e-4, betas=(0.9, 0.95))
trainer.train()
```

`get_optimizer(model, lr, weight_decay, optimizer_type)` из `llm.training.optimizer` создаёт `"adamw"`, `"adam"` (L2-регуляризация, а не decoupled decay) или `"sgd"` (момент 0.9) с теми же группами weight decay.

### Mixtral: load-balancing loss

У Mixtral с `router_aux_loss_coef > 0` в конфиге (в HF — `0.001`) `Trainer` прибавляет к loss вспомогательный loss роутера, который выравнивает загрузку экспертов; паддинг в его статистику не входит. В `evaluate()` он не прибавляется. Без него эксперты часто перестают использоваться — см. главу [Mixture-of-Experts](../textbook/mixture-of-experts.md).

## Чего в Trainer нет

`Trainer` — минимальный учебный цикл. В нём нет:

- сохранения чекпоинтов и лучшей модели — сохраняйте сами, `model.save(path)` (см. [Сохранение и загрузку](checkpoints.md));
- продолжения обучения: состояние оптимизатора и планировщика не сохраняется, новый `Trainer` начнёт с нулевых моментов Adam и снова с warmup;
- смешанной точности (AMP), накопления градиентов, распределённого обучения;
- логирования метрик кроме `loss_history` и печати.

Если это нужно, пишите свой цикл: модели — обычные `nn.Module`, `forward` возвращает `(logits, cache)`, а loss — `trainer.compute_lm_loss(logits, labels)` или тот же расчёт вручную.

## Скрипт экспериментов

`experiments/llm_only/run_llm_experiment.py` обучает и запускает любую из шести моделей на учебном корпусе по JSON-конфигу:

```bash
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action train --config experiments/llm_only/configs/llama_train.json
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action generate --config experiments/llm_only/configs/llama_generate.json
```

`train` обучает BPE-токенизатор (или загружает готовый из `checkpoints/bpe_tokenizer.json`), создаёт модель по `model_config`, обучает её `Trainer` с параметрами из раздела `training` (`learning_rate`, `batch_size`, `num_epochs`, `warmup_ratio` или `warmup_steps`) и сохраняет веса и конфиг в `checkpoints/`. Валидацию скрипт не запускает. Формат конфигов — в [experiments/README.md](../../experiments/README.md).

## Если что-то не так

- **Loss не падает** — проверьте реальный learning rate (`trainer.optimizer.param_groups[0]["lr"]`) и длину warmup относительно числа шагов.
- **Начальный loss сильно больше `ln V`** (V — размер словаря) — проблема инициализации или масштаба логитов; у всех шести моделей свежий loss около `ln V`.
- **Loss подозрительно быстро падает** — проверьте, что паддинг не входит в loss: доля меток, отличных от `-100`, в батче должна соответствовать длине текстов.
- **Loss стал NaN** — слишком большой learning rate, отсутствие warmup (особенно у post-LN GPT-1) или переполнение float16.

Подробная диагностика с примерами — в главе [Обучение](../textbook/training.md#диагностика).
