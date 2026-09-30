# Установка

[Оглавление](README.md) · [Модели и конфиги →](models.md)

## Требования

- **Python 3.10+.**
- **[uv](https://docs.astral.sh/uv/)** — менеджер пакетов: репозиторий — uv workspace из двух пакетов.
- **PyTorch.** Корневой проект закрепляет `torch==2.8.0`; сама библиотека `llm` требует `torch>=2.3.0` и `numpy>=1.24.0`, других зависимостей у неё нет.
- **Transformers и Datasets** нужны только пакету `hf-proxy` и для загрузки весов HuggingFace.

Видеокарта не обязательна: всё работает на CPU, а `Trainer` сам выбирает `cuda`, если она доступна.

## Установка

```bash
git clone https://github.com/pese-git/llm-arch-research.git
cd llm-arch-research
uv sync                 # llm, hf-proxy и зависимости корневого проекта
uv sync --extra dev     # плюс pytest, ruff, black, mypy, jupyter
```

`uv sync` ставит оба пакета workspace в режиме редактирования: правки в `llm/src/` видны без переустановки.

| Пакет | Каталог | Импорт | Зависимости |
|---|---|---|---|
| `llm` | `llm/src/llm` | `import llm` | `torch`, `numpy` |
| `hf-proxy` | `hf-proxy/src/hf_proxy` | `import hf_proxy` | `llm`, `transformers`, `datasets` |

Скрипты и ноутбуки запускайте через `uv run` из корня репозитория: пути в конфигах экспериментов (`checkpoints/...`) относительные.

## Проверка

```bash
uv run python -c "
import torch
from llm.models.llama import Llama
model = Llama({'vocab_size': 100, 'embed_dim': 32, 'num_heads': 4, 'num_layers': 2,
               'max_position_embeddings': 64, 'dropout': 0.0})
print(model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5, do_sample=False))
"
```

Команда печатает тензор из 8 id: 3 токена промпта и 5 сгенерированных. Полный набор тестов — `uv run pytest` из корня (около 1000 тестов, меньше минуты на CPU; подробнее — в [Тестах](../dev/testing.md)).

## Что дальше

- [Модели и конфиги](models.md) — как собрать модель нужной архитектуры.
- [Обучение](training.md) — первый запуск обучения на учебном корпусе.
- [Ноутбуки](../../notebooks/) — пошаговый разбор каждой архитектуры в Jupyter (нужен `uv sync --extra dev`).
