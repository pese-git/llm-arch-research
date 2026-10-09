# План: реальное обучение моделей в библиотеке `llm`

Цель: довести обучающий контур библиотеки от «15 предложений на CPU» до обучения
моделей 10–100M параметров на корпусах 10⁷–10⁸ токенов на одном GPU/MPS, с чекпоинтами,
продолжением обучения и измеримым качеством. Каждая фаза самодостаточна: её можно
выполнять в новом контексте, опираясь только на этот файл и указанные в нём источники.

Порядок: фазы 1–2 дают минимально рабочее обучение, 3–4 — скорость, 5 — масштаб, 6 — проверка.
Каждая фаза — отдельная ветка и PR по правилам [docs/dev/conventions.md](../docs/dev/conventions.md).

Архитектура и обоснование решений — [docs/dev/training-design.md](../docs/dev/training-design.md) и [docs/dev/decisions.md](../docs/dev/decisions.md) (ADR-001…007).

Статус-строки по мере выполнения: «Статус: PR #…».

---

## Фаза 0. Факты и разрешённые API

Собрано чтением кода и проверкой в `.venv` (torch 2.8.0, numpy 2.3.3). Реализации
должны использовать только перечисленное; API, которых здесь нет, перед использованием
проверять в `.venv`, а не предполагать.

### 0.1. Что уже есть в проекте

| Что | Где | Факт |
|---|---|---|
| Цикл обучения | `llm/src/llm/training/trainer.py` | `Trainer(model, train_dataset, val_dataset=None, lr=3e-4, batch_size=8, num_epochs=3, warmup_steps=None, warmup_ratio=None)`; методы `train()`, `evaluate() -> float`, `compute_lm_loss(logits, labels)`, `_forward(batch)`, `num_warmup_steps(n)`; атрибуты `loss_history`, `train_loader`, `val_loader`, `optimizer`, `scheduler`, `device`, `warmup_steps`, `warmup_ratio`. Устройство: строка 124, `cuda` либо `cpu`, MPS не рассматривается. Обучение только по эпохам, валидация в конце эпохи, чекпоинтов нет |
| Оптимизатор | `training/optimizer.py` | `get_optimizer(model, lr, weight_decay=0.01, optimizer_type="adamw")`, `weight_decay_param_groups(model, wd)` (decay только для `p.dim() >= 2`) |
| Расписание | `training/scheduler.py` | `get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps) -> LambdaLR` |
| Loss-метки | `datasets/lm_example.py` | `IGNORE_INDEX = -100`; `lm_example(token_ids, block_size, pad_token_id) -> {"input_ids","attention_mask","labels"}` |
| Датасеты | `datasets/text_dataset.py`, `streaming_text_dataset.py`, `text_with_special_tokens_dataset.py` | принимают `List[str]`; одна строка = один пример, обрезка по `block_size`, правый паддинг. `datasets/__init__.py` пуст, импорт всегда полным путём `from llm.datasets.<module> import ...` |
| Модель | `core/base_model.py` | `forward(x, use_cache=False, cache=None, attention_mask=None) -> (logits, cache)`; `save(path)` пишет `{"model_class","config","state_dict"}`; `load(path, device="cpu")` читает с `weights_only=True`, проверяет класс, возвращает `.eval()`; `auxiliary_loss() -> Optional[Tensor]`; `max_seq_len` — свойство из `self._max_seq_len` |
| **Порядок аргументов GPT** | `models/gpt/gpt.py:156-158` | `forward(self, x, attention_mask=None, use_cache=False, cache=None)` — `attention_mask` **второй позиционный**. Все вызовы моделей делать только именованными аргументами |
| Цикл по слоям | `llama.py:171-181`, `gpt.py:206-216`, `gpt2.py:200-210`, `mistral.py:159`, `mixtral.py:257`, `gemma.py:241` | `for i, decoder in enumerate(self._decoders): decoder(out, use_cache=..., cache=..., padding=padding)`; декодер возвращает `(result, kv)` или `(result, None)` |
| MHA | `core/multi_head_attention.py` | `__init__(num_heads, emb_size, head_size, max_seq_len, rope=None, dropout=0.1, attention_dropout=0.0, bias=True)`; `forward(x, use_cache=True, cache=None, padding=None)`; scores строка 244, причинная маска 249 (`self._tril_mask[start:start+T, :start+T]`), паддинг 250–252 (`padding.apply(...)` → `[B,1,T,T_kv]`, True = разрешено), `masked_fill(~mask, -inf)` 253, softmax + attention dropout 256; кэш — пара `(k, v)` |
| GQA | `core/group_query_attention.py` | `__init__(num_q_heads, num_kv_heads, emb_size, head_size, max_seq_len, window_size=None, rope=None, dropout=0.1, bias=True)` — **без** `attention_dropout`; `forward(x, use_cache=True, cache=None, padding=None)`; scores 257, маска окна 263–270, softmax 273; кэш — **тройка** `(k, v, next_pos)`; `_repeat_kv_heads` 291–351; маска окна `_create_sliding_window_mask(max_seq_len, window_size, device=None)` 353–410 |
| Паддинг | `core/padding.py` | `Padding(NamedTuple)`: `key_mask [B, start+T] bool`, `positions [B, T] long`; `Padding.apply(allowed, start_pos, key_start)`; `padding_from_attention_mask(attention_mask, x, start_pos=0) -> Optional[Padding]` (None, если маска None или вся из единиц) |
| Токенизатор | `tokenizers/base_tokenizer.py` | `BaseTokenizer(ABC)`: абстрактные `train(texts, vocab_size=1000, **kw)`, `encode(text, **kw) -> List[int]`, `decode(tokens, **kw) -> str`; конкретные `get_vocab_size()`, `add_special_tokens`, `save(filepath)` (JSON: `vocab, vocab_size, pad_token, unk_token, bos_token, eos_token, tokenizer_type`), `load(filepath)` (вызывает `cls()` без аргументов), `__len__`; атрибуты `pad_token_id/unk_token_id/bos_token_id/eos_token_id: Optional[int]` |
| Скрипт экспериментов | `experiments/llm_only/run_llm_experiment.py` | `train`: `TextDataset(train_texts, tokenizer, block_size=model_config["max_position_embeddings"])`, `Trainer(...)`, `torch.save(model.state_dict(), ...)` + JSON конфига. Валидацию не запускает. Корпус — `TRAIN_TEXTS` из `experiments/shared/configs.py:6-22`, 15 строк |
| Ноутбуки | `notebooks/{gpt,gpt2,llama,mistral,mixtral,gemma}.ipynb` | вызывают `Trainer(model, train_dataset, val_dataset, lr=..., batch_size=..., num_epochs=..., warmup_ratio=...)`, затем `trainer.train()`, `trainer.evaluate()`, читают `trainer.loss_history`; `gemma.ipynb` ещё `Trainer(m, train_dataset, num_epochs=1)` и `trainer.compute_lm_loss(...)`. **Эти сигнатуры менять нельзя**, только расширять |
| Тесты Trainer | `llm/tests/training/test_trainer.py` | 19 тестов; хелперы `ToyLMDataset` (без `attention_mask`), `TinyModel` (возвращает голый тензор), `TupleModel`, `CharTokenizer`, `GPT_CONFIG`, `TEXTS` |
| Эталонные тесты | `llm/tests/core/test_moe.py:75-93`, `test_load_balancing_loss.py:27-47` | образец: наивная реализация + `torch.allclose(..., atol=...)` |
| Фикстуры | `llm/tests/conftest.py` | `device` (cuda/cpu), `batch_size=2`, `seq_len=64`, `vocab_size=1000`, `embed_dim=256` |
| CI | `.github/workflows/tests.yml` | `uv sync --frozen --extra dev --python 3.12`, `uv run pytest -q -rs`; **падает, если тест скипнут с «could not import»** — новые зависимости для тестов либо в `dev`-extras, либо тесты без них. Изменения `llm/src/**` запускают ещё `figures-check.yml` (2.5 мин) и `notebooks-run.yml` (7 ноутбуков, nbconvert) |

### 0.2. Разрешённые API PyTorch и NumPy (проверены в `.venv`)

```
torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None,
    dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> Tensor
    # attn_mask bool: True = разрешено; float: прибавляется к scores. enable_gqa — torch >= 2.5
torch.autocast(device_type: str, dtype=None, enabled=True, cache_enabled=None)
    # device_type="mps" с torch.bfloat16 работает в torch 2.8 (проверено: matmul даёт bf16)
torch.amp.GradScaler(device="cuda", ...)   # только для float16 на CUDA
torch.utils.checkpoint.checkpoint(function, *args, use_reentrant: Optional[bool] = None, **kwargs)
    # всегда use_reentrant=False; kwargs пробрасываются в function
torch.optim.AdamW(params, lr, weight_decay, fused: Optional[bool] = None)
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
torch.backends.mps.is_available() -> bool;  torch.mps.synchronize()
torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=0, pin_memory=False, drop_last=False)
torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
torch.distributed.init_process_group(backend); torch.nn.parallel.DistributedDataParallel(model)
torch.Generator().manual_seed(n); torch.get_rng_state() / set_rng_state()
numpy.memmap(filename, dtype=np.uint16, mode="r")   # mode "w+" с shape для записи
```

### 0.3. Правила проекта, которые действуют на все фазы

- Зависимости `llm` — только `torch` и `numpy` (conventions.md:19). HF/tiktoken — только через утиную типизацию или в `hf-proxy`.
- Докстринги и комментарии по-русски; формулы в `r"""…"""` (проверяется `tests/test_source.py`).
- Поведение по умолчанию не меняется: новый ключ конфига или аргумент сохраняет старое поведение (conventions.md:22). Плохой конфиг → `ValueError` в конструкторе (:23).
- Тесты: результаты, а не формы; эталон или число, посчитанное руками; `torch.manual_seed`, `dropout: 0.0` (testing.md:36-43).
- Документация в том же PR: `docs/guide/` + таблица ключей в `llm/README.md`; механизмы — в главу учебника; несовместимости — в `CHANGELOG.md` (группы «Результат модели», «Чекпоинты и конфиги», «API», новые записи в начало группы, формат `- **Суть** ([#N](url)): …`).
- У каждой страницы `docs/` под `# Заголовком` стоит `<!-- description: … -->` до 160 символов.
- Числа в документации — из реальных запусков; примеры кода должны запускаться.
- `black` + `ruff`; `uv run pytest` из корня зелёный перед PR.

### Анти-паттерны (общие)

- Не передавать `attention_mask` в `MultiHeadAttention`/`GroupedQueryAttention`: они принимают `padding: Padding`.
- Не предполагать у GQA `attention_dropout`, у кэша GQA — пару: там тройка.
- Не вызывать модели позиционно: у `GPT` второй аргумент — `attention_mask`.
- Не добавлять `transformers`, `tokenizers`, `tiktoken`, `datasets` в зависимости `llm`.
- Не ломать вызовы `Trainer` из ноутбуков и 19 существующих тестов.
- Не писать tqdm-прогресс в stdout ноутбуков (CI `notebooks/tools/check.py` падает на прогресс-барах — ноутбуки уже перенаправляют stdout, новые print-ы должны идти туда же).
- Не менять `TRAIN_TEXTS` и существующие `*_train.json`: на них завязаны ноутбуки и `figures-check`.

---

## Фаза 1. Конвейер данных: непрерывные блоки токенов из файла

Ветка `feat/token-block-dataset`. Статус: PR [#80](https://github.com/pese-git/llm-arch-research/pull/80), слит. Отличие от плана: `len(TokenBlockDataset) = num_tokens // block_size` (сдвиг делает loss, лишний токен не нужен).

### Что реализовать

1. **`llm/src/llm/datasets/token_block_dataset.py`** — класс `TokenBlockDataset(torch.utils.data.Dataset)`:
   - `__init__(self, tokens, block_size: int)`, где `tokens` — путь к `.bin` (читается `np.memmap(path, dtype, mode="r")`, dtype выводится из соседнего `<path>.json` или аргумента `dtype`) либо готовый `np.ndarray`/`list[int]`.
   - `__len__ = (len(tokens) - 1) // block_size` — последний неполный блок отбрасывается.
   - `__getitem__(i)` возвращает `{"input_ids": long[block_size], "labels": long[block_size]}` для среза `tokens[i*block_size : i*block_size + block_size]`; `labels` — те же токены (сдвиг делает `Trainer.compute_lm_loss`, как у остальных датасетов, см. `docs/guide/data.md:60`). Без `attention_mask`: `Trainer._forward` передаёт её только если есть в батче.
   - `ValueError` в конструкторе, если `block_size < 1` или токенов меньше `block_size + 1`.
   - Докстринг объясняет, почему блоки непрерывны и паддинга нет (нарезка потока как в GPT-2 / nanoGPT), и что перемешивание блоков делает `DataLoader(shuffle=True)`.
2. **`llm/src/llm/datasets/tokenize_corpus.py`** — функция `tokenize_file(text_path, tokenizer, out_path, *, eos_token_id=None, dtype=None, chunk_lines=10_000) -> int`:
   - читает текст построчно, `tokenizer.encode(line, add_special_tokens=False)` (так вызывают существующие датасеты), между документами (пустая строка) вставляет `eos_token_id`, если задан;
   - пишет `np.memmap(out_path, dtype, mode="w+", shape=(n,))` чанками; dtype `uint16`, если `tokenizer.get_vocab_size() <= 65535`, иначе `uint32`; рядом `<out_path>.json` с `{"dtype": ..., "num_tokens": ..., "vocab_size": ...}`;
   - возвращает число токенов.
3. **`llm/src/llm/evaluation/perplexity.py`** (сейчас пустой): `lm_loss(model, loader, device, max_batches=None) -> float` и `perplexity(model, loader, device, max_batches=None) -> float = exp(lm_loss)`. Loss считать той же функцией, что и `Trainer`: вынести тело `Trainer.compute_lm_loss` в `llm/training/loss.py::causal_lm_loss(logits, labels)`, а метод оставить обёрткой (тесты `test_compute_lm_loss_*` зовут метод). `llm/evaluation/__init__.py` экспортирует `perplexity`. Пустые `benchmark.py` и `utils.py` удалить.
4. **Скрипт подготовки корпуса** `experiments/shared/prepare_corpus.py`: `--input <txt>` (или `--url` для скачивания через `urllib`), `--val-ratio 0.01`, `--tokenizer <json|"train">`, `--vocab-size`, `--out data/<name>/` → `train.bin`, `val.bin`, `tokenizer.json`. Сетевых зависимостей нет. README в `experiments/` описывает, где взять корпус (например, TinyStories в виде одного txt, или дамп русской Википедии), и что `data/` в `.gitignore`.
5. **`run_llm_experiment.py`**: если в конфиге есть секция `"data": {"train": "data/x/train.bin", "val": "data/x/val.bin", "tokenizer": "data/x/tokenizer.json"}`, использовать `TokenBlockDataset` и передавать `val_dataset`; иначе — старый путь через `TRAIN_TEXTS` без изменений. Новый конфиг-образец `experiments/llm_only/configs/llama_corpus_train.json` (модель ~20M: `embed_dim 384, num_heads 6, num_layers 6, max_position_embeddings 256`).

### Источники для копирования

- Форма возвращаемого словаря и `IGNORE_INDEX`: `llm/src/llm/datasets/lm_example.py`.
- Стиль докстринга датасета (назначение, аргументы, пример, ссылки): `datasets/text_dataset.py:9-50`.
- Как `Trainer` обрабатывает батч без `attention_mask`: `training/trainer.py::_forward`.
- Тесты датасета: `llm/tests/datasets/test_text_dataset.py` (структура), `test_dataset_contents.py` (проверка содержимого).

### Проверка

- `llm/tests/datasets/test_token_block_dataset.py`: длина = `(n-1)//block`; `input_ids == labels`; соседние блоки стыкуются без пропусков (блок `i` заканчивается на токене `i*block+block-1`); dtype `torch.long`; чтение из `.bin` даёт те же тензоры, что из списка; `ValueError` на коротком входе.
- `llm/tests/datasets/test_tokenize_corpus.py`: во `tmp_path` маленький текст → `.bin` + `.json`; число токенов совпадает с ручным `encode`; `eos` стоит между документами; `uint16` при малом словаре.
- `llm/tests/evaluation/test_perplexity.py` (новая папка; в `llm/tests/` подпапки без `__init__.py`, только корневой `llm/tests/__init__.py`): для модели с равномерными логитами perplexity ≈ `V`; `perplexity == exp(Trainer.evaluate())` на одном и том же наборе (с `dropout 0.0`).
- `uv run pytest` зелёный; `uv run python experiments/llm_only/run_llm_experiment.py --model gpt --action train --config experiments/llm_only/configs/gpt_train.json` работает как раньше.
- Документация: `docs/guide/data.md` (новый раздел «Корпус из файла»), `docs/guide/training.md` (perplexity), `docs/dev/architecture.md` (дерево модулей: `evaluation/` больше не «пуста»), `llm/README.md`, `experiments/README.md` (секция `data`), `docs/guide/limitations.md` (убрать пункты «датасеты не склеивают тексты», «оценки качества нет»).

### Анти-паттерны фазы

- Не токенизировать в `__getitem__` и не хранить список Python-int в памяти для больших корпусов: только `np.memmap`.
- Не добавлять паддинг и `attention_mask` в `TokenBlockDataset`.
- Не сдвигать `labels` в датасете (сдвиг делает loss).
- Не менять `TextDataset` и `TRAIN_TEXTS`.

---

## Фаза 2. Trainer: устройство, шаги, чекпоинты, продолжение, логи

Ветка `feat/trainer-checkpoints`. Статус: PR [#81](https://github.com/pese-git/llm-arch-research/pull/81). Все новые аргументы — именованные, с умолчаниями, воспроизводящими текущее поведение.

### Что реализовать

1. **Устройство.** Новый аргумент `device: str | torch.device | None = None`. При `None` — `cuda` → `mps` (`torch.backends.mps.is_available()`) → `cpu`. Это меняет поведение на Mac (раньше CPU) → запись в `CHANGELOG.md` группа «API».
2. **Обучение по шагам.** `max_steps: int | None = None`. Если задан, цикл идёт по шагам, `num_epochs` игнорируется, `DataLoader` перезапускается при исчерпании; `total_steps` для расписания = `max_steps`. Если не задан — старое поведение по эпохам, `loss_history` по эпохам сохраняется как есть.
3. **Валидация по интервалу.** `eval_interval: int | None = None` (шагов), `eval_batches: int | None = None` (ограничение числа батчей валидации). При `None` — как сейчас, в конце эпохи. `evaluate()` по-прежнему возвращает `float` и оставляет модель в `eval`; после валидации внутри `train()` вернуть `model.train()`.
4. **Чекпоинты.** `checkpoint_dir: str | None = None`, `save_interval: int | None = None`, `keep_best: bool = True`.
   - `save_checkpoint(path)` пишет через `torch.save` словарь `{"model_class", "config", "state_dict"}` (ровно как `BaseModel.save`, чтобы файл читался `BaseModel.load` с `weights_only=True`) **плюс** `"optimizer"`, `"scheduler"`, `"step"`, `"epoch"`, `"loss_history"`, `"log"`, `"rng"` (`torch.get_rng_state()`), `"trainer_version": 1`. Папку создаёт `os.makedirs(..., exist_ok=True)`.
   - `last.pt` каждые `save_interval` шагов и в конце; `best.pt` при улучшении валидационного loss, если `keep_best` и есть `val_dataset`.
   - `load_checkpoint(path)` / `Trainer.resume(path, ...)`: восстанавливает веса, оптимизатор, планировщик (планировщик создаётся до загрузки, с теми же `total_steps`), шаг, историю, RNG. Проверить `model_class` как в `BaseModel.load`.
   - `BaseModel.load` должен продолжать читать такой файл: он требует лишь `{"config","state_dict"}` и проверяет `model_class`. Добавить тест.
5. **Логи.** `self.log: list[dict]` с записями `{"step", "epoch", "lr", "train_loss", "val_loss"}`; `log_path: str | None = None` → JSON после каждой валидации и в конце. `print` оставить, `tqdm` оставить (ноутбуки его перенаправляют).
6. **Воспроизводимость.** `seed: int | None = None` → `torch.manual_seed` и `generator` для `DataLoader(shuffle=True)`.
7. **`run_llm_experiment.py`**: пробрасывать новые ключи секции `training` (`max_steps`, `eval_interval`, `eval_batches`, `save_interval`, `checkpoint_dir`, `device`, `seed`), флаг `--resume <path>`. Сохранение модели перевести на `model.save(config["model_weights"])` (сейчас сохраняется голый `state_dict`) — несовместимость для `generate`: читать обоими способами, запись в CHANGELOG «Чекпоинты и конфиги».

### Источники для копирования

- Формат файла модели и проверки при загрузке: `core/base_model.py:80-138`.
- Валидация аргументов и `ValueError` в конструкторе: `trainer.py:101-109`.
- Как `scheduler` создаётся в `train()` по `total_steps`: `trainer.py::train` (начало).
- Текст про отсутствие resume, который нужно заменить: `docs/guide/checkpoints.md:48-50`, `docs/textbook/training.md:742-753`, `docs/guide/training.md` «Чего в Trainer нет».

### Проверка

- Существующие 19 тестов `test_trainer.py` проходят без правок.
- Новые тесты в `test_trainer.py` или `test_trainer_checkpoints.py`:
  - `device=None` на машине без CUDA/MPS даёт `cpu`; явный `device="cpu"` принимается; неизвестное устройство → `ValueError`.
  - `max_steps=7` на датасете из 3 батчей делает ровно 7 шагов (`scheduler.last_epoch == 7`), `lr` в конце 0.
  - `save_interval=2, max_steps=4, checkpoint_dir=tmp_path` → `last.pt` существует; `BaseModel.load(last.pt)` возвращает модель с теми же логитами.
  - **Resume-эквивалентность**: обучение 6 шагов с `seed` ≡ обучение 3 шагов → `save` → новый `Trainer` → `resume` → 3 шага; веса совпадают `atol=1e-6` (`dropout 0.0`, CPU). Это главный тест фазы.
  - `best.pt` обновляется только при уменьшении val loss.
  - `log_path` содержит записи с `step`, `lr`, `val_loss`.
- Ноутбуки: `cd notebooks && jupyter nbconvert --execute --to notebook --inplace gpt.ipynb` проходит (поведение по умолчанию не изменилось).
- Документация: `docs/guide/training.md` (таблица параметров, раздел «Чекпоинты и продолжение»), `docs/guide/checkpoints.md` «Продолжение обучения» переписать, `docs/textbook/training.md:648-753` обновить, `docs/guide/limitations.md` убрать пункты про чекпоинты/resume/логи, `experiments/README.md` новые ключи `training`, `CHANGELOG.md` две записи.

### Анти-паттерны фазы

- Не менять формат `BaseModel.save` и не ломать чтение `weights_only=True` (никаких несериализуемых объектов в чекпоинте).
- Не делать `num_epochs`/`warmup` обязательными и не менять умолчание `warmup_steps=100`.
- Не сохранять чекпоинт из `evaluate()`; не вызывать `evaluate()` без `val_loader`.
- Не переносить логику выбора устройства в модели.

---

## Фаза 3. SDPA-путь в attention с тестом равенства

Ветка `perf/sdpa-attention`. Ручная реализация остаётся эталоном и умолчанием.

### Что реализовать

1. **`MultiHeadAttention`** и **`GroupedQueryAttention`**: аргумент конструктора `attention_impl: str = "manual"` (`"manual" | "sdpa"`, иначе `ValueError`), атрибут `self._attention_impl`. В `forward` после вычисления `q, k, v` (и конкатенации кэша, и `_repeat_kv_heads` для GQA):
   - `manual` — существующий код без изменений;
   - `sdpa` — построить ту же булеву маску, что и сейчас (`causal_mask`/`window_mask` после `padding.apply`), и вызвать `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p, scale=1/sqrt(head_size))`, где `p = self._attn_dropout.p if self.training else 0.0` для MHA и `0.0` для GQA (у неё нет attention dropout). Маска `[T, T_kv]` без паддинга → расширить до `[1, 1, T, T_kv]`; с паддингом она уже `[B, 1, T, T_kv]`. Быстрый случай `is_causal=True` без `attn_mask` — только когда `padding is None`, `cache is None` и `window_size is None` (для MHA — всегда при тех условиях).
   - `MultiQueryAttention` — учебный модуль, не трогать.
2. **Пробросить ключ конфига** `attention_impl` (умолчание `"manual"`) через все шесть моделей и пять декодеров: `CachedDecoder`, `GptDecoder`, `Gpt2Decoder`, `MistralDecoder`, `MixtralDecoder`, `GemmaDecoder` → аргумент `attention_impl` в конструкторе → в `MultiHeadAttention`/`GroupedQueryAttention`. Модели читают `config.get("attention_impl", "manual")`; `ValueError` на неизвестное значение в конструкторе модели.
3. **Вспомогательная функция** `llm/core/attention_impl.py::set_attention_impl(model, impl)` — проходит `model.modules()` и переключает `_attention_impl` у уже построенной модели (удобно для сравнения в тестах и для загруженных чекпоинтов).

### Источники для копирования

- Точки вставки: MHA `multi_head_attention.py:244-259`, GQA `group_query_attention.py:257-274`.
- Семантика булевой маски и `Padding.apply`: `core/padding.py:38-56`.
- Как ключ конфига проходит модель → декодер → блок: `models/llama/llama.py:106-126`, `core/cached_decoder.py:63-74`.
- Образец эталонного теста: `llm/tests/core/test_moe.py:75-93`.

### Проверка

- `llm/tests/core/test_attention_impl.py`, параметризованно по `(MHA, GQA)` × `{без паддинга, с паддингом слева, с паддингом в середине}` × `{без кэша, с кэшем в 2 шага}` × для GQA `{window None, window 4}`: при `torch.manual_seed(0)`, `dropout=0.0` выходы `manual` и `sdpa` совпадают `atol=1e-5`; градиенты по входу совпадают `atol=1e-5`.
- `test_model_contract.py`: добавить параметр `attention_impl` в `MODELS`-прогон или отдельный тест, что все шесть моделей с `attention_impl="sdpa"` дают те же логиты, что с `"manual"`, и что `generate` с KV-кэшем совпадает.
- HF-parity тесты (`llm/tests/models/test_*_hf_parity.py`) проходят с `attention_impl="sdpa"` (параметризовать фикстуру).
- Замер и запись в документацию: `tok/s` manual vs sdpa на MPS/CPU для конфига 20M из фазы 1 (число — из реального запуска, по правилу conventions.md:41).
- Документация: ключ в `llm/README.md` и `docs/guide/models.md`; `docs/guide/limitations.md` — пункт «нет оптимизированных ядер» переписать; глава учебника `docs/textbook/attention.md` (или где описан MHA) — абзац «Реализация: ручная и SDPA»; `figures-check` остаётся зелёным (умолчание не изменилось).

### Анти-паттерны фазы

- Не использовать `is_causal=True` вместе с `attn_mask` и не использовать его при `start_pos > 0`.
- Не передавать float-маску с `-inf` там, где нужна булева (SDPA трактует bool как «True = разрешено», а `~mask` из ручного пути — наоборот).
- Не включать `enable_gqa=True` без проверки `torch.__version__ >= 2.5` (библиотека обещает `torch>=2.3`); по умолчанию повторять KV-головы как сейчас.
- Не менять умолчание на `"sdpa"` в этой фазе.

---

## Фаза 4. Mixed precision, накопление градиентов, косинусное расписание, DataLoader

Ветка `perf/trainer-amp`.

### Что реализовать

1. **`dtype: str = "float32"`** в `Trainer` (`"float32" | "bfloat16" | "float16"`). При не-float32: `torch.autocast(device_type=self.device.type, dtype=...)` вокруг forward+loss; для `"float16"` на CUDA — `torch.amp.GradScaler`: `scaler.scale(loss).backward()`, `scaler.unscale_(optimizer)` перед `clip_grad_norm_`, `scaler.step`, `scaler.update`. `"float16"` на CPU/MPS → `ValueError` (нет GradScaler). Веса остаются fp32 (см. докстринг `core/feed_forward.py:115-118`).
2. **`grad_accum_steps: int = 1`**: loss делится на `grad_accum_steps`, `optimizer.step()`/`scheduler.step()`/`zero_grad` раз в `grad_accum_steps` микробатчей; «шаг» в `max_steps`, `eval_interval`, `save_interval` — это шаг оптимизатора. `clip_grad_norm_` — перед `optimizer.step`.
3. **`max_grad_norm: float = 1.0`** вместо захардкоженного `1.0`; `None` отключает.
4. **Косинусное расписание**: `training/scheduler.py::get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0) -> LambdaLR`, формула в докстринге (`r"""`): после warmup `lr = min + (1-min)·0.5·(1+cos(π·progress))`. В `Trainer` аргумент `lr_schedule: str = "linear"` (`"linear" | "cosine"`).
5. **DataLoader**: `num_workers: int = 0`, `pin_memory: bool | None = None` (`True` на CUDA по умолчанию), `drop_last=True` при обучении по шагам.
6. **Оптимизатор**: в `get_optimizer` — `fused=True` для AdamW на CUDA (проверить `"fused" in signature`), иначе как сейчас.

### Источники для копирования

- Паттерн `LambdaLR` и стиль докстринга расписания: `training/scheduler.py`.
- Тест, что autocast не меняет dtype весов: `llm/tests/core/test_feed_forward.py:228-234`.
- Тесты расписания: `llm/tests/training/test_scheduler.py`, `test_trainer.py:261-273` (`scheduler.lr_lambdas[0](k)`).
- Глава учебника про mixed precision и косинус: `docs/textbook/training.md:426-434, 593-616`.

### Проверка

- `test_scheduler.py`: косинус — значения в точках `0`, `warmup`, середина, конец, посчитанные руками; `min_lr_ratio` соблюдается.
- `test_trainer.py`: **эквивалентность накопления** — `batch_size=4, grad_accum_steps=1` и `batch_size=2, grad_accum_steps=2` (одинаковый порядок данных через `seed`, `dropout 0.0`, один шаг) дают одинаковые веса `atol=1e-6`.
- `dtype="bfloat16"` на CPU: loss конечен, `p.dtype == torch.float32` у всех параметров после обучения. `dtype="float16"` на CPU → `ValueError`.
- `max_grad_norm=None`: градиенты не обрезаются (норма до/после одинакова).
- Замер на MPS для 20M-модели: `tok/s` fp32 vs bf16, в документацию.
- Документация: `docs/guide/training.md` таблица параметров; `docs/textbook/training.md` раздел «Trainer» (что включает AMP и накопление) и «Mixed precision» (ссылка на реализацию); `docs/guide/limitations.md` убрать «нет AMP, накопления градиентов»; `experiments/README.md` ключи.

### Анти-паттерны фазы

- Не вызывать `scaler.step` без предварительного `unscale_` при clipping.
- Не ставить `autocast` вокруг `optimizer.step()`.
- Не считать `loss.item()` внутри накопления умноженным на `1/grad_accum` при печати: логировать невзвешенный loss микробатча или сумму за шаг, одинаково в `loss_history`.
- Не менять умолчание `lr_schedule` на `cosine`.

---

## Фаза 5. Масштаб: внешний токенизатор, gradient checkpointing, DDP

Ветка(и) `feat/external-tokenizer`, `perf/gradient-checkpointing`, `feat/ddp` — три отдельных PR.

### 5.1. Адаптер внешнего токенизатора

- `llm/src/llm/tokenizers/external_tokenizer.py::ExternalTokenizer(BaseTokenizer)`: оборачивает любой объект с `encode(str) -> list[int]` и `decode(list[int]) -> str` (HF `tokenizers.Tokenizer` — у него `encode(...).ids`; `tiktoken.Encoding` — `encode`/`decode`; передавать через два callable `encode_fn`, `decode_fn` плюс `vocab_size` и id спецтокенов). `train()` → `NotImplementedError` с подсказкой. `save/load` — сохраняют только метаданные и имя внешнего токенизатора (`tokenizer_type`), не сам словарь; в докстринге объяснить, что внешний объект надо пересоздать самому. Экспорт в `tokenizers/__init__.py`.
- `tokenize_corpus.py` принимает такой адаптер без изменений (использует только `encode` и `get_vocab_size`).
- Тесты `llm/tests/tokenizers/test_external_tokenizer.py` **без** импорта `tokenizers`/`tiktoken` (CI падает на «could not import»): фейковый объект с `encode`/`decode`; проверить `encode/decode` round-trip, `get_vocab_size`, спецтокены, `train()` → `NotImplementedError`. Опциональный тест с настоящим `tokenizers` — только если пакет добавлен в `dev`-extras корневого `pyproject.toml`; иначе не писать.
- Документация: `docs/guide/data.md` раздел «Внешний токенизатор», `docs/guide/limitations.md` абзац про символьный BPE дополнить.

### 5.2. Gradient checkpointing

- Ключ конфига `gradient_checkpointing: bool = False` у всех шести моделей. В цикле по слоям, когда `self.training and self._gradient_checkpointing and not use_cache`:
  `decoder_result = torch.utils.checkpoint.checkpoint(decoder, out, use_cache=False, cache=None, padding=padding, use_reentrant=False)`; иначе прямой вызов. Точки вставки: `llama.py:172-174`, `gpt.py:207-209`, `gpt2.py:201-203`, `mistral.py:159`, `mixtral.py:257`, `gemma.py:241`.
- Mixtral: `auxiliary_loss()` (`mixtral.py:277-291`) берёт `decoder._ff.router_logits`, которые MoE сохраняет в `self.router_logits` на последнем forward (`core/moe.py:187`). Внутри `checkpoint` первый проход идёт без графа, поэтому эти логиты не связаны с параметрами роутера, и aux-loss перестанет давать градиент. Решение: для Mixtral оборачивать в `checkpoint` функцию, которая возвращает `(out, router_logits)`, и класть возвращённые логиты обратно в `decoder._ff.router_logits`. Это главный риск подзадачи; тест на равенство градиентов роутера обязателен.
- Тесты: в `test_model_contract.py` параметризованный тест — логиты и градиенты всех параметров совпадают с `gradient_checkpointing` и без (`atol=1e-6`, `dropout 0.0`); для Mixtral — `auxiliary_loss()` одинаков. `eval()` + `use_cache=True` не использует checkpointing (generate работает).
- Документация: `llm/README.md` таблица ключей, `docs/guide/models.md`, `docs/textbook/training.md` раздел «Память при обучении» (строка 617) — абзац о checkpointing со ссылкой на ключ.

### 5.3. Распределённое обучение (DDP)

- Не встраивать в `Trainer` ветвление на каждом шаге. Сделать `experiments/llm_only/train_ddp.py` для `torchrun --nproc_per_node=N`: `init_process_group("nccl"|"gloo")`, `device = cuda:{local_rank}`, модель → `DistributedDataParallel`, `DistributedSampler` для `TokenBlockDataset`, `Trainer` с `device=` и готовым `train_loader` (добавить в `Trainer` опциональные `train_loader`/`val_loader`, чтобы подменить `DataLoader`), `auxiliary_loss` через `model.module`, чекпоинты/логи только с `rank == 0` (аргумент `Trainer(is_main_process=...)`), `sampler.set_epoch(epoch)`.
- Проверка: тест-смоук в `llm/tests/training/test_ddp_smoke.py`, который запускает `torchrun --nproc_per_node=1 experiments/llm_only/train_ddp.py --config <tiny>` через `subprocess` с `backend=gloo` и таймаутом 120 с, и проверяет, что появился `last.pt`. Многопроцессный запуск — ручная проверка, результат (loss, tok/s, число GPU) записать в `docs/guide/training.md`.
- Документация: `docs/guide/training.md` раздел «Несколько GPU», `docs/guide/limitations.md` переписать раздел «Обучение» целиком под новое состояние, `docs/dev/architecture.md` дерево (`training/loss.py`, `evaluation/perplexity.py`, `datasets/token_block_dataset.py`, `tokenize_corpus.py`, `tokenizers/external_tokenizer.py`).

### Анти-паттерны фазы 5

- Не делать `tokenizers`/`tiktoken` зависимостью `llm` и не писать тесты с `importorskip` без добавления пакета в `dev`-extras.
- Не использовать `use_reentrant=True` и не применять checkpointing при `use_cache=True`.
- Не сохранять `state_dict` DDP-обёртки с префиксом `module.`: сохранять `model.module`.

---

## Фаза 6. Итоговая проверка

1. `uv run pytest -q -rs` из корня — зелёный, без скипов «could not import».
2. `uv run ruff check llm hf-proxy experiments docs/tools && uv run black --check llm hf-proxy experiments docs/tools`.
3. `uv run python docs/tools/figures.py --check` — иллюстрации не изменились (умолчания сохранены).
4. Ноутбуки: для каждого `cd notebooks && jupyter nbconvert --execute --to notebook --inplace <name>.ipynb && python tools/check.py` (так делает `notebooks-run.yml`).
5. Grep анти-паттернов (ожидается пусто):
   ```
   grep -rn "use_reentrant=True" llm/src
   grep -rn "is_causal=True" llm/src | grep attn_mask
   grep -rnE "import (transformers|tokenizers|tiktoken|datasets)" llm/src
   grep -rn "torch.save(model.state_dict()" experiments
   ```
6. **Реальный прогон**, результат в `docs/guide/training.md` (числа только из запуска): модель 20M (`llama_corpus_train.json`), корпус ≥ 10M токенов, `attention_impl="sdpa"`, `dtype="bfloat16"`, `max_steps` на ~1 эпоху; зафиксировать `tok/s`, итоговую perplexity на `val.bin`, время, устройство. Прервать на середине и продолжить через `--resume` — кривая loss должна продолжиться без скачка (приложить `log.json`).
7. Обновить `docs/guide/limitations.md` под фактическое состояние и закрыть пункты в `docs/dev/backlog.md`, если они там заведены.
