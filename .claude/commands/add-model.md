---
description: Добавить архитектуру по docs/dev/adding-model.md — блок декодера, модель, перенос весов HF, тесты и сверка, эксперименты, практикум, глава пособия, сайт
argument-hint: <архитектура, например Qwen2 или Phi-3>
---
Архитектура: $ARGUMENTS

Это `/change` для частого случая. Выполни шаги команды `/change` со следующими уточнениями.

На шаге 1 **research-architect** обязан: найти статью и эталон в `transformers`, перечислить отличия от ближайшей существующей модели (образец — `Llama` или `Mistral`), проверить, какие механизмы уже есть в `core/` и какие нужны новые, и чем модель будет сверяться (`convert_hf_state_dict`, совпадение логитов `atol=1e-4` и greedy-генерации).

Обязательный состав результата (проверь в итоге):
- новые механизмы — модули `llm/src/llm/core/` со своими тестами в `llm/tests/core/` (сверка с формулой);
- `llm/src/llm/models/<name>/` — модель по контракту `BaseModel`, `__init__.py`, `hf_weights.py`;
- модель во всех параметризованных тестах и `MODELS` в `test_model_contract.py`; `tests/models/test_<name>.py`; сверка с HF (`test_<name>_hf_parity.py`) — с подтверждением, что тесты ловят ошибку (verification-engineer);
- ветка в `load_model_class()` в `experiments/llm_only/run_llm_experiment.py`, конфиги `<name>_train.json` и `<name>_generate.json` (промпты из символов корпуса, `warmup_ratio`), дымовой прогон обучения и генерации;
- по желанию пользователя — `notebooks/<name>.ipynb` по шаблону;
- глава `docs/textbook/<name>.md` по шаблону части II, оглавление и таблица в `docs/textbook/README.md`, карточка в `site/src/landing/index.mdx`, рецепт в `docs/guide/hf-weights.md`, ключи в `docs/guide/models.md` и `llm/README.md`;
- расхождения со статьёй, найденные по ходу, — в бэклог.
