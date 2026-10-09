# Обучение
<!-- description: Обучение моделей библиотеки llm: Trainer, обучение по шагам, чекпоинты и продолжение, оптимизатор, расписание learning rate, скрипт экспериментов. -->

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
        warmup_steps=None, warmup_ratio=None, *,
        device=None, max_steps=None, eval_interval=None, eval_batches=None,
        checkpoint_dir=None, save_interval=None, keep_best=True, log_path=None, seed=None)
```

| Параметр | Что задаёт |
|---|---|
| `lr` | пиковый learning rate |
| `batch_size`, `num_epochs` | размер батча и число эпох; обучающие данные перемешиваются каждую эпоху |
| `warmup_steps` / `warmup_ratio` | длина линейного warmup: числом шагов или долей от всех шагов, `ceil(N_steps · warmup_ratio)`, как в HF. Оба сразу — `ValueError`; ни одного — 100 шагов |
| `val_dataset` | если задан, после каждой эпохи (или каждые `eval_interval` шагов) вызывается `evaluate()` |
| `device` | `"cuda"`, `"mps"`, `"cpu"` или `torch.device`; `None` — `cuda`, если доступна, иначе `cpu`; `"auto"` — первое доступное из `cuda`, `mps`, `cpu` (на Apple Silicon в несколько раз быстрее CPU). Недоступное устройство — `ValueError` |
| `max_steps` | обучение по шагам: ровно столько шагов оптимизатора, данные идут по кругу, `num_epochs` не используется; расписание learning rate считается на `max_steps` |
| `eval_interval`, `eval_batches` | валидация каждые `eval_interval` шагов (и в конце обучения) вместо конца эпохи; `eval_batches` ограничивает её первыми N батчами |
| `checkpoint_dir`, `save_interval`, `keep_best` | папка чекпоинтов: `last.pt` каждые `save_interval` шагов и в конце обучения, `best.pt` при улучшении валидационного loss. См. [Чекпоинты и продолжение](#чекпоинты-и-продолжение) |
| `log_path` | JSON с записями `{"step", "epoch", "lr", "train_loss", "val_loss"}` после каждой валидации и каждой эпохи; то же — в `trainer.state.log` |
| `seed` | `torch.manual_seed` и порядок батчей: перестановка эпохи k зависит только от `seed + k`, поэтому после `resume` она та же |

Шаг — это шаг оптимизатора. Всё новое выключено по умолчанию: `Trainer(model, train)` ведёт себя как раньше. MPS не включается сам: ноутбуки и иллюстрации учебника подают модели CPU-тензоры, и модель на MPS их бы не приняла.

Что делает `train()`:

- **Оптимизатор** — AdamW, `weight_decay=0.01` только на матрицах (веса `Linear` и эмбеддинги); bias и веса нормализаций не затухают. Бета — значения PyTorch по умолчанию, `(0.9, 0.999)`.
- **Расписание** — линейный warmup от 0 до `lr`, затем линейный спад до 0 к концу обучения. Если warmup не короче всего обучения, `train()` предупреждает: learning rate не дойдёт до заданного.
- **Loss** — cross-entropy следующего токена: логиты сдвигаются на одну позицию относительно меток, позиции с меткой `-100` не учитываются. Батч без единой цели даёт loss 0, а не NaN.
- **Gradient clipping** — общая норма градиента обрезается до 1.0.
- **`attention_mask`** из батча передаётся в модель, если она есть.
- **Устройство** — `cuda`, если доступна, иначе `cpu`; `device="auto"` добавляет `mps`; модель переносится туда в конструкторе.

`evaluate()` переводит модель в режим eval, считает средний loss по валидационному набору без градиентов и возвращает его. Перплексия — `llm.evaluation.perplexity(model, loader, device)`: `exp` от того же среднего loss (`lm_loss` возвращает сам loss; `max_batches` ограничивает оценку частью набора). После `train()` модель остаётся в режиме eval, если была валидация, и в режиме train — если не было; перед генерацией вызывайте `model.eval()`.

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

## Чекпоинты и продолжение

```python
trainer = Trainer(model, train, val, lr=3e-4, batch_size=16, max_steps=5000, warmup_ratio=0.05,
                  eval_interval=500, eval_batches=50,
                  checkpoint_dir="checkpoints/run", save_interval=500,
                  log_path="checkpoints/run/log.json", seed=0)
trainer.train()
print(trainer.state.step, trainer.state.best_val_loss, trainer.state.log[-1])
```

`checkpoints/run/last.pt` пишется каждые `save_interval` шагов и в конце, `best.pt` — когда валидационный loss улучшился. Файл — надмножество формата `model.save`: `{"model_class", "config", "state_dict"}` плюс секция `trainer` с состоянием оптимизатора, планировщика, шагом, историей и генератором случайных чисел. Поэтому `GPT.load("checkpoints/run/last.pt")` загружает его как обычную модель, а `trainer.resume(path)` — как состояние обучения:

```python
trainer = Trainer(model, train, val, lr=3e-4, batch_size=16, max_steps=5000, warmup_ratio=0.05, seed=0)
trainer.resume("checkpoints/run/last.pt")   # веса, Adam, расписание, шаг, история, RNG
trainer.train()                              # продолжает с сохранённого шага
```

`resume` требует тот же класс модели и те же аргументы расписания (`lr`, `batch_size`, `num_epochs`, `max_steps`, `warmup_steps`, `warmup_ratio`), иначе `ValueError` с перечнем различий: с другими значениями число шагов и learning rate не совпали бы с сохранённым планировщиком. С `seed` и `dropout: 0.0` на CPU продолженное обучение побитово совпадает с непрерывным — это проверяет тест `test_resume_is_equivalent_to_uninterrupted_training`; на GPU — с точностью ядер. Чекпоинт посреди эпохи тоже продолжается с того же батча: уже пройденные батчи эпохи пропускаются.

`trainer.save_checkpoint(path)` сохраняет вручную в любой момент. Формат файла — в [Сохранении и загрузке](checkpoints.md#продолжение-обучения).

## Чего в Trainer нет

`Trainer` — учебный цикл: он читается целиком. В нём нет:

- смешанной точности (AMP), накопления градиентов, распределённого обучения;
- логирования во внешние системы: только `state.log`, JSON и печать.

Если это нужно, пишите свой цикл: модели — обычные `nn.Module`, `forward` возвращает `(logits, cache)`, а loss — `trainer.compute_lm_loss(logits, labels)` или тот же расчёт вручную.

## Скрипт экспериментов

`experiments/llm_only/run_llm_experiment.py` обучает и запускает любую из шести моделей по JSON-конфигу — на учебном корпусе или на [корпусе из файла](data.md#корпус-из-файла), если в конфиге есть секция `data` (пример — `llama_corpus_train.json`):

```bash
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action train --config experiments/llm_only/configs/llama_train.json
uv run python experiments/llm_only/run_llm_experiment.py --model llama --action generate --config experiments/llm_only/configs/llama_generate.json
```

`train` обучает BPE-токенизатор (или загружает готовый из `checkpoints/bpe_tokenizer.json`), создаёт модель по `model_config`, обучает её `Trainer` с параметрами из раздела `training` (`learning_rate`, `batch_size`, `num_epochs` или `max_steps`, `warmup_ratio` или `warmup_steps`, а также необязательные `device`, `eval_interval`, `eval_batches`, `checkpoint_dir`, `save_interval`, `keep_best`, `seed`, `train_log_path`) и сохраняет модель через `model.save` в `model_weights`. `--resume checkpoints/<run>/last.pt` продолжает обучение с чекпоинта. С секцией `data` токенизатор и блоки берутся из `data/<name>/`, после обучения печатаются валидационный loss и перплексия; без неё валидации нет. Формат конфигов — в [experiments/README.md](../../experiments/README.md).

## Если что-то не так

- **Loss не падает** — проверьте реальный learning rate (`trainer.optimizer.param_groups[0]["lr"]`) и длину warmup относительно числа шагов.
- **Начальный loss сильно больше `ln V`** (V — размер словаря) — проблема инициализации или масштаба логитов; у всех шести моделей свежий loss около `ln V`.
- **Loss подозрительно быстро падает** — проверьте, что паддинг не входит в loss: доля меток, отличных от `-100`, в батче должна соответствовать длине текстов.
- **Loss стал NaN** — слишком большой learning rate, отсутствие warmup (особенно у post-LN GPT-1) или переполнение float16.

Подробная диагностика с примерами — в главе [Обучение](../textbook/training.md#диагностика).
