# LLM Architecture Research — инструкции для Claude Code

Учебная реализация «с нуля» на PyTorch шести архитектур — GPT, GPT-2, LLaMA, Mistral, Mixtral, Gemma — с учебным пособием, практикумами и проверяемым результатом. Полные правила — в `docs/dev/` (читай нужную страницу перед работой):

- `docs/dev/architecture.md` — пакеты, модули `llm`, контракт `BaseModel`, поток данных;
- `docs/dev/adding-model.md` — этапы добавления архитектуры;
- `docs/dev/testing.md` — как тестировать (сверка с эталоном, инварианты, регрессия);
- `docs/dev/conventions.md` — ветки, коммиты, PR, документация, CHANGELOG, бэклог;
- `docs/dev/decisions.md` (ADR), `docs/dev/training-design.md`, `plans/01-real-training.md` — обучение;
- `docs/dev/backlog.md` — журнал расхождений со статьями (большой: читай нужный пункт, а не целиком).

## Главное
- **Результат проверяется.** Модель сверяется с HuggingFace на тех же весах (`atol=1e-4`, совпадение greedy-генерации), блок — с наивной реализацией формулы. Тест проверяет результат, а не форму; регрессионный тест **падает на старом коде** — это проверяется откатом исправления.
- **Зависимости `llm` — только `torch` и `numpy`.** HuggingFace — в `hf-proxy` или в тестах через `pytest.importorskip`.
- **Совместимость по умолчанию.** Новый ключ конфига, меняющий форму весов или результат, по умолчанию сохраняет прежнее поведение. Неверный конфиг — `ValueError` в конструкторе.
- API torch/numpy, которого нет в коде проекта, проверяй в `.venv` (torch 2.8.0), а не предполагай.
- Докстринги и комментарии — на русском; формулы в докстрингах — в сырых строках `r"""…"""` (проверяет `tests/test_source.py`).
- Числа в документации получены запуском; пособие описывает текущее поведение, история — в CHANGELOG и бэклоге.
- `black`/`ruff` — только для изменённых файлов: репозиторий целиком их не проходит, CI их не запускает; не переформатируй чужой код в рамках задачи.
- Иллюстрации правятся только через `docs/tools/figures.py`; схемы ноутбуков — через `notebooks/tools/diagrams.py`; выводы ноутбуков после CI не коммитятся.

## Команды
```bash
uv sync --frozen --extra dev
uv run pytest -q -rs
uv run python docs/tools/figures.py --check      # ~2,5 мин
python notebooks/tools/diagrams.py --check && python notebooks/tools/check.py
cd site && npm ci && npm run build
```

## Процесс
Ветка от `master` с префиксом `fix/`, `feat/`, `refactor/`, `perf/`, `test/`, `docs/`, `chore/`; один PR — одна задача, вливается merge-коммитом. Коммиты — на английском, Conventional Commits, тело объясняет «почему». Описание PR — на русском: «Проблема», «Что сделано», «Несовместимость», «Проверка». Без явной просьбы: не делать `git commit`/`push`, не открывать PR.

## Команда субагентов (`.claude/agents/`)
| Агент | Зона | Пишет код |
|---|---|---|
| `research-architect` | план, ADR, пункт бэклога, задачи | `plans/`, `docs/dev/` |
| `model-engineer` | `llm/core`, `llm/models`, перенос весов HF, `hf-proxy` | да |
| `training-engineer` | `training`, `datasets`, `tokenizers`, `evaluation`, `experiments/` | да |
| `verification-engineer` | сверка с HF и формулами, инварианты, регрессия «падает на старом коде» | тесты |
| `notebook-engineer` | `notebooks/`, `docs/tools/figures.py`, данные бенчмарка | ноутбуки, рисунки |
| `docs-author` | пособие, руководство, `llm/README.md`, CHANGELOG, бэклог, сайт | документы |
| `check-runner` | повтор CI | нет |
| `paper-reviewer` | соответствие статьям и эталону | нет |
| `code-reviewer` | соглашения, тесты, документация, PR | нет |

Команды: `/change`, `/backlog-fix`, `/add-model`, `/plan-phase`, `/review`, `/check`, `/figures`.
