# Добавление модели
<!-- description: Как добавить архитектуру: блок декодера, модель, конфиги экспериментов, сверка с HuggingFace и глава пособия. -->

[← Устройство репозитория](architecture.md) · [Оглавление](README.md) · [Тесты →](testing.md)

Новая архитектура проходит те же этапы, что и шесть существующих. Удобный образец — самая близкая по устройству модель: для pre-LN модели с RoPE это `Llama` (`models/llama/llama.py`), для GQA со скользящим окном — `Mistral`.

## 1. Блок декодера

Соберите блок из модулей `core/`. Если он укладывается в pre-LN схему «норма → attention → residual → норма → FFN → residual», используйте `CachedDecoder`, передав `norm_layer` и `feed_forward_layer`; иначе напишите свой `core/<name>_decoder.py` по образцу `mistral_decoder.py`.

Блок принимает `(x, use_cache, cache, padding)` и передаёт `padding` в attention. Новый механизм (другая нормализация, активация, вариант attention) — отдельный модуль в `core/` со своими тестами в `tests/core/`.

## 2. Класс модели

`models/<name>/<name>.py` — наследник `BaseModel`. По [контракту](architecture.md#контракт-basemodel):

- конструктор: `super().__init__(config)`, `self._max_seq_len`, `head_size = resolve_head_size(config, ...)` (проверки делимости и чётности для RoPE), слои, в конце — `self.apply(partial(init_normal_, std=config.get("initializer_range", DEFAULT_INITIALIZER_RANGE)))`;
- `forward(x, use_cache=False, cache=None, attention_mask=None) -> (logits, cache)` с `cache_start_pos`, `check_sequence_length` и `padding_from_attention_mask`;
- `generate` наследуется — не переопределяйте его.

**Необязательные ключи — с прежним поведением по умолчанию.** Всё, что меняет форму весов (`bias`, `intermediate_size`, `tie_word_embeddings`), по умолчанию сохраняет структуру, с которой созданы существующие чекпоинты; конфиг оригинала включается ключами. Неверный конфиг — `ValueError` в конструкторе, а не падение в `forward`.

Экспорт — `models/<name>/__init__.py` (класс и, если есть, `convert_hf_state_dict`).

## 3. Перенос весов HuggingFace

Если у архитектуры есть модель в `transformers`, добавьте `models/<name>/hf_weights.py` с `convert_hf_state_dict(hf_state_dict, ...)`: переименование ключей, незнакомый ключ — `KeyError`, для RoPE на соседних парах — перестановка строк `q_proj`/`k_proj` (`_hf_to_meta_rows` из `models/llama/hf_weights.py`).

## 4. Тесты

Минимальный набор (подробнее — [Тесты](testing.md)):

- добавьте модель в словарь `MODELS` в `tests/models/test_model_contract.py` — общие проверки позиций, лимита длины и конфига;
- `tests/models/test_<name>.py` — формы выхода, кэш (префилл кусками и генерация с кэшем совпадают с полным `forward`), конфиг;
- добавьте модель в тесты, которые перебирают все модели: `test_attention_mask.py`, `test_kv_cache.py`, `test_save_load.py`, `test_state_dict.py`, `test_generate_args.py`, `test_llama_family_init.py` (или аналог инициализации);
- сверка с HuggingFace — случайная модель `transformers` той же конфигурации, `convert_hf_state_dict`, совпадение логитов (`atol=1e-4`) и greedy-генерации с KV-кэшем. Обычно это отдельный файл `tests/models/test_<name>_hf_parity.py` (`test_llama_hf_parity.py`, `test_gemma_hf_parity.py`); близкие модели можно сверять в общем файле, как Mistral и Mixtral в `test_mistral_mixtral_hf_parity.py`, а GPT-1 и GPT-2 сверяются в `test_gpt_weight_tying.py`.

## 5. Эксперименты

- Ветка в `load_model_class()` в `experiments/llm_only/run_llm_experiment.py`.
- Конфиги `experiments/llm_only/configs/<name>_train.json` и `<name>_generate.json`: промпты — из символов учебного корпуса (иначе они кодируются в `<unk>`), `warmup_ratio` вместо фиксированного `warmup_steps`.
- По желанию — практикум `notebooks/<name>.ipynb` по шаблону из [notebooks/README.md](../../notebooks/README.md).

## 6. Документация

- Глава в `docs/textbook/<name>.md` по общему шаблону глав части II — разделы `##` в таком порядке:
  1. «Что вы узнаете», «Предварительные знания»;
  2. «Обзор» (внутри — `###` «Научный вклад») и, если нужно, «Изменения относительно <предыдущей модели>»;
  3. «Архитектура блока декодера» (схема Mermaid), «Прямой проход в формулах»;
  4. разделы о механизмах, которые появились в этой модели (например, «Grouped Query Attention» в Mistral);
  5. «Компоненты», «Разбор кода», «Подсчёт параметров», «Конфигурация»;
  6. «Загрузка весов HuggingFace», «Отличия от оригинала», «Генерация», затем дополнительные разделы о модели (например, «LLaMA 2 и GQA»);
  7. «Типичные ошибки и тонкости», «Что изменилось в <следующей модели>»;
  8. «Итоги», «Вопросы и упражнения», «Литература».

  Добавьте её в оглавление и таблицу архитектур `docs/textbook/README.md` — по оглавлению строится меню сайта.
- Карточка модели на визитке сайта — `site/src/landing/index.mdx`, раздел «Шесть архитектур».
- Рецепт загрузки весов — в [docs/guide/hf-weights.md](../guide/hf-weights.md), ключи конфига — в [docs/guide/models.md](../guide/models.md) и таблице ключей в `llm/README.md`.
- Расхождения со статьёй, найденные по ходу, — в [бэклог](backlog.md).
