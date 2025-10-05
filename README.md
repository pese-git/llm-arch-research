# LLM Architecture Research

Исследовательский проект для разработки и обучения архитектур больших языковых моделей (LLM).

## 🏗️ Архитектура проекта

Проект организован как монорепозиторий с использованием **uv** workspace:

- **`llm`** — основная библиотека с реализацией архитектур LLM (GPT, GPT-2)
- **`hf-proxy`** — адаптер для интеграции с HuggingFace
- **`experiments`** — скрипты обучения и экспериментов
- **`notebooks`** — исследовательские ноутбуки

## 📁 Структура проекта

```
llm-arch-research/
│
├── pyproject.toml        # корневой workspace конфиг
├── uv.lock
│
├── llm/                  # основная библиотека архитектур
│   ├── pyproject.toml
│   └── src/llm/
│       ├── core/         # базовые компоненты
│       │   ├── base_model.py
│       │   ├── decoder.py
│       │   ├── multi_head_attention.py
│       │   ├── head_attention.py
│       │   ├── feed_forward.py
│       │   ├── token_embeddings.py
│       │   └── positional_embeddings.py
│       ├── models/gpt/   # GPT и GPT-2 реализация
│       │   ├── gpt.py
│       │   ├── gpt2.py
│       │   └── __init__.py
│       ├── training/     # утилиты обучения
│       │   ├── dataset.py
│       │   ├── trainer.py
│       │   ├── optimizer.py
│       │   └── scheduler.py
│       ├── evaluation/   # оценка моделей
│       └── tokenizers/   # токенизаторы
│           ├── base_tokenizer.py
│           └── bpe_tokenizer.py
│
├── hf-proxy/             # адаптер HuggingFace
│   ├── pyproject.toml
│   └── src/hf_proxy/
│       ├── hf_config.py
│       ├── hf_adapter.py
│       ├── hf_tokenizer.py
│       └── hf_utils.py
│
├── experiments/          # скрипты обучения и экспериментов
│   ├── hf_integration/   # интеграция с HuggingFace
│   │   ├── train_with_hf_trainer.py
│   │   ├── generate_with_hf_tools.py
│   │   ├── simple_hf_training.py
│   │   └── test_hf_proxy.py
│   ├── llm_only/         # обучение без HF
│   │   ├── train_gpt_bpe.py
│   │   └── generate_gpt_bpe.py
│   └── shared/           # общие утилиты
│       ├── configs.py
│       └── data.py
│
├── checkpoints/          # сохраненные модели и токенизаторы
└── notebooks/            # исследовательские ноутбуки
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Установка всех зависимостей workspace
uv sync

# Установка с dev-зависимостями
uv sync --extra dev
```

### Запуск обучения GPT

```bash
# Обучение базовой GPT модели
uv run python experiments/llm_only/train_gpt_bpe.py

# Обучение с интеграцией HuggingFace
uv run python experiments/hf_integration/simple_hf_training.py
```

### Тестирование hf-proxy

```bash
# Базовое тестирование интеграции
uv run python experiments/hf_integration/test_hf_proxy.py

# Генерация через HF инструменты
uv run python experiments/hf_integration/generate_with_hf_tools.py
```

### Использование в коде

```python
from llm.models.gpt import GPT, GPT2
from llm.tokenizers import BPETokenizer
from hf_proxy import HFAdapter, HFTokenizerAdapter

# Создание GPT модели
config = {
    "vocab_size": 50257,
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 4,
    "max_position_embeddings": 128,
    "dropout": 0.1
}
model = GPT(config)

# Создание GPT-2 модели (пример)
gpt2_config = {
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "max_position_embeddings": 1024,
    "dropout": 0.1
}
gpt2_model = GPT2(gpt2_config)

# Генерация текста
generated = model.generate(
    input_ids, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7
)

# Использование с HuggingFace через hf-proxy
hf_model = HFAdapter.from_llm_model(model)
hf_tokenizer = HFTokenizerAdapter(tokenizer)

# Генерация через HF интерфейс
generated = hf_model.generate(
    input_ids=inputs['input_ids'],
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)
```

## 🛠️ Технологический стек

- **Python 3.10+** — язык программирования
- **uv** — современный менеджер пакетов и workspace
- **PyTorch 2.8+** — фреймворк глубокого обучения
- **Transformers** — интеграция с HuggingFace
- **Datasets** — работа с данными
- **TOML** — конфигурационные файлы

## 📦 Зависимости

### Корневой workspace
```toml
[project]
dependencies = ["tqdm>=4,<5"]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=24.0.0", 
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "jupyter>=1.0.0",
]
test = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
]
```

### Пакет llm
```toml
[project]
dependencies = [
    "torch>=2.3.0",
    "numpy>=1.24.0",
]
```

### Пакет hf-proxy
```toml
[project]
dependencies = [
    "torch>=2.3.0",
    "transformers>=4.44.0",
    "datasets>=2.20.0",
]
```

## 🎯 Реализованные возможности

### Архитектуры GPT и GPT-2
- ✅ Токенные и позиционные эмбеддинги
- ✅ Многоголовое внимание с causal mask
- ✅ Декодерные блоки с residual connections
- ✅ Layer normalization
- ✅ Dropout регуляризация
- ✅ Отдельные реализации GPT и GPT-2 (различия в масштабе и деталях архитектуры)

### Генерация текста
- ✅ Жадный поиск (greedy decoding)
- ✅ Вероятностное сэмплирование
- ✅ Top-k сэмплирование
- ✅ Nucleus sampling (top-p)
- ✅ Контроль температуры

### Обучение
- ✅ Датасет для языкового моделирования
- ✅ Базовый тренировочный цикл
- ✅ Оптимизатор AdamW
- ✅ Сохранение чекпоинтов

### Интеграция с HuggingFace (hf-proxy)
- ✅ Адаптер моделей для совместимости с HF интерфейсами
- ✅ Адаптер токенизаторов с поддержкой всех методов HF
- ✅ Сохранение и загрузка в HF формате
- ✅ Совместимость с HF Trainer и pipelines
- ✅ Генерация через стандартные HF интерфейсы

## 🔬 Эксперименты с hf-proxy

### Успешно протестированные функции:

1. **Базовая интеграция** (`test_hf_proxy.py`)
   - ✅ Создание HF адаптера для токенизаторов
   - ✅ Создание HF адаптера для моделей
   - ✅ Токенизация и декодирование
   - ✅ Forward pass через адаптированную модель
   - ✅ Сохранение и загрузка моделей

2. **Упрощенное обучение** (`simple_hf_training.py`)
   - ✅ Обучение GPT модели с использованием hf-proxy
   - ✅ Ручной цикл обучения без сложных зависимостей
   - ✅ Сохранение результатов обучения

3. **Генерация через HF инструменты** (`generate_with_hf_tools.py`)
   - ✅ Загрузка моделей в HF формате
   - ✅ Генерация через стандартные HF интерфейсы
   - ✅ Сравнение стратегий генерации
   - ✅ Интерактивная генерация

### Решенные проблемы:

- ✅ Исправление метода `pad` в токенизаторе для обработки разных типов данных
- ✅ Корректная загрузка моделей с передачей конфигурации
- ✅ Совместимость с HF экосистемой

## 📊 Примеры работы

### Обучение модели
```bash
🚀 УПРОЩЕННОЕ ОБУЧЕНИЕ GPT С HF-PROXY
=========================================================
🔧 Подготовка данных...
📊 Данные: 10 train, 2 validation
🔧 Подготовка токенизатора...
✅ Токенизатор создан (vocab_size=473)
🔧 Подготовка модели...
✅ Модель создана
🎯 Обучение модели...
📊 Результаты обучения:
   Final train loss: 4.6802
   Final val loss: 5.1834
✅ Модель сохранена
```

### Генерация через HF интерфейсы
```bash
🧪 Тестирование HuggingFace pipeline...
🎯 Генерация текста через HF адаптер
🔤 Промпт: 'Искусственный'
🎯 Результат: 'Искусственный интеллект продолжает развиваться...'
```

## 🔧 Разработка

### Добавление зависимостей
```bash
# В корневой проект
uv add package-name

# В конкретный пакет
cd llm && uv add package-name

# Dev-зависимости
uv add --dev pytest black
```

### Запуск тестов
```bash
uv run pytest
```

### Форматирование кода
```bash
uv run black .
uv run ruff check .
```

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте feature ветку
3. Внесите изменения
4. Запустите тесты: `uv run pytest`
5. Отформатируйте код: `uv run black . && uv run ruff check .`
6. Создайте pull request

## 📄 Лицензия

MIT License

---

**Разработано с ❤️ для исследований в области LLM**

*Обновлено: Октябрь 2025*
