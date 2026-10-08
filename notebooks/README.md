# Практикумы

Ноутбуки дополняют [учебное пособие](../docs/textbook/README.md): теория и разбор кода остаются в главах, а здесь механизмы пишутся руками, сверяются с библиотекой `llm` и исследуются на обученной модели.

| Ноутбук | Глава | Что пишется руками |
|---|---|---|
| [bpe.ipynb](bpe.ipynb) | [Токенизация](../docs/textbook/tokenization.md) | обучение BPE, кодирование по слияниям и жадное |
| [gpt.ipynb](gpt.ipynb) | [GPT-1](../docs/textbook/gpt.md) | голова attention, multi-head как список голов, FFN, LayerNorm, post-LN блок |
| [gpt2.ipynb](gpt2.ipynb) | [GPT-2](../docs/textbook/gpt2.md) | pre-LN блок, масштаб инициализации, температура, top-k, top-p |
| [llama.ipynb](llama.ipynb) | [LLaMA](../docs/textbook/llama.md) | RMSNorm, SwiGLU, RoPE, attention с RoPE и KV-кэшем |
| [mistral.ipynb](mistral.ipynb) | [Mistral](../docs/textbook/mistral.md) | grouped query attention, скользящее окно, кэш, ограниченный окном |
| [mixtral.ipynb](mixtral.ipynb) | [Mixtral](../docs/textbook/mixtral.md) | Mixture-of-Experts, load-balancing loss |
| [gemma.ipynb](gemma.ipynb) | [Gemma](../docs/textbook/gemma.md) | GeGLU, multi-query attention, RMSNorm в параметризации Gemma |

## Шаблон

Каждый ноутбук архитектуры устроен одинаково.

1. **Пишем сами.** Только то, что в этой архитектуре новое по сравнению с предыдущей, по формулам из главы. Имена полей совпадают с библиотечными, чтобы веса переносились через `state_dict`.
2. **Сверяем с библиотекой.** Тот же модуль из `llm.core` на тех же весах: полный проход, генерация с кэшем, префилл кусками. Сверка численная, через `torch.allclose`.
3. **Собираем модель.** Блок декодера и модель из своих и библиотечных блоков, наследуя `generate`, `save` и `load` от `BaseModel`. Логиты и жадная генерация сверяются с `llm.models.*`.
4. **Обучаем.** Конфиг из `experiments/llm_only/configs/<model>_train.json`, корпус из `experiments/shared`, `BPETokenizer`, датасет с `<eos>` и `Trainer` из библиотеки.
5. **Смотрим внутрь.** Карты внимания, кэш, рецептивное поле, распределение по экспертам, масштаб residual-потока: то, чего нет в тексте главы.
6. **Упражнения.** Задачи для этого ноутбука плюс ссылка на вопросы главы с ответами.

## Схемы

Схемы в ноутбуках те же, что в главах учебника: Mermaid-блоки из `docs/textbook/*.md`. GitHub не рендерит Mermaid внутри ipynb и вырезает HTML вроде `<details>`, поэтому в ячейку вкладывается PNG с подписью-ссылкой на главу, где лежит исходник. Схему задаёт маркер в markdown-ячейке, картинку и подпись генерирует [`tools/diagrams.py`](tools/diagrams.py):

```markdown
<!-- diagram: textbook/mistral.md | Архитектура Mistral -->
```

```bash
uv run python notebooks/tools/diagrams.py          # обновить схемы, изменившиеся в docs
uv run python notebooks/tools/diagrams.py --check  # проверить, что все схемы актуальны
```

Рендер идёт через mermaid-cli (`npx -y -p @mermaid-js/mermaid-cli mmdc`), нужны Node.js и при первом запуске сеть. Не правьте сгенерированный блок между маркером и `<!-- /diagram -->` руками: при следующем запуске он будет перезаписан.

## Проверки в CI

Ноутбуки хранятся с выводами, и два workflow следят, чтобы они не разошлись с учебником и библиотекой.

| Workflow | Когда | Что проверяет |
|---|---|---|
| [`notebooks-check.yml`](../.github/workflows/notebooks-check.yml) | PR и `master`, где меняются `notebooks/` или `docs/textbook/` | Секунды, без torch. Схемы совпадают с учебником (`tools/diagrams.py --check`). [`tools/check.py`](tools/check.py): файл проходит `nbformat.validate`, в выводах нет ошибок, трассировок и полос tqdm, счётчики выполнения идут подряд (ноутбук перезапущен целиком), размер не больше 600 КиБ |
| [`notebooks-run.yml`](../.github/workflows/notebooks-run.yml) | PR и `master`, где меняются `notebooks/`, `llm/src/`, `experiments/shared/`, конфиги `llm_only` или `uv.lock` | Каждый ноутбук выполняется целиком на CPU, семь параллельных задач; сам ноутбук считается 10–80 секунд, остальное — установка зависимостей (torch с CUDA-колёсами, дальше берётся из кэша). `assert` на совпадение с библиотекой ловит расхождения, когда меняется `llm/src`. К свежему выводу применяется тот же `check.py`, выполненный ноутбук лежит в артефакте задачи 7 дней |

Выводы выполненного в CI ноутбука **не коммитятся**: числа зависят от платформы. Если проверка упала, запустите то же локально:

```bash
uv run python notebooks/tools/check.py                    # выводы
uv run python notebooks/tools/diagrams.py                 # перерисовать устаревшие схемы
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace mistral.ipynb   # перезапустить ноутбук
```

## Запуск

Из корня репозитория:

```bash
uv sync --extra dev
uv run jupyter lab notebooks/mistral.ipynb
```

Все ячейки выполняются на CPU за одну-две минуты на ноутбук. Корень репозитория ноутбуки находят сами по `pyproject.toml`, поэтому запускать можно из любой папки.

Прогнать все ноутбуки целиком и обновить выводы:

```bash
cd notebooks && for n in bpe gpt gpt2 llama mistral mixtral gemma; do uv run jupyter nbconvert --to notebook --execute --inplace "$n.ipynb"; done
```

Ноутбуки хранятся с выводами, чтобы читать их на GitHub без запуска. После правок перезапускайте ноутбук целиком («Restart Kernel and Run All Cells»), чтобы выводы соответствовали коду.
