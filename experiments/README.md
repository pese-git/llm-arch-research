# Эксперименты с LLM архитектурами

Унифицированная структура экспериментов для обучения и генерации текста моделями LLM.

## 📁 Структура экспериментов

```
experiments/
├── llm_only/                    # Эксперименты только с библиотекой llm
│   ├── train_gpt_bpe.py         # Обучение GPT с BPE токенизатором
│   └── generate_gpt_bpe.py      # Генерация с GPT + BPE
├── hf_integration/              # Эксперименты с hf-proxy
│   ├── train_with_hf_trainer.py # Обучение через HF Trainer
│   └── generate_with_hf_tools.py # Генерация через HF инструменты
├── shared/                      # Общие утилиты
│   ├── data.py                  # Загрузка и подготовка данных
│   └── configs.py               # Конфигурации моделей
└── README.md                    # Этот файл
```

## 🚀 Быстрый старт

### 1. Только библиотека llm (автономный режим)

```bash
# Обучение GPT модели с собственным BPE токенизатором
uv run python experiments/llm_only/train_gpt_bpe.py

# Генерация текста обученной моделью
uv run python experiments/llm_only/generate_gpt_bpe.py
```

### 2. Интеграция с HuggingFace через hf-proxy

```bash
# Обучение через HuggingFace Trainer
uv run python experiments/hf_integration/train_with_hf_trainer.py

# Генерация через HF инструменты
uv run python experiments/hf_integration/generate_with_hf_tools.py
```

## 📊 Сравнение подходов

| Аспект | Только llm | С hf-proxy |
|--------|------------|------------|
| **Зависимости** | Только PyTorch | + HuggingFace Transformers |
| **Обучение** | Собственный Trainer | HF Trainer |
| **Генерация** | Прямой вызов модели | HF pipeline & интерфейсы |
| **Гибкость** | Полный контроль | Совместимость с HF экосистемой |
| **Сложность** | Проще | Более сложная настройка |

## 🔧 Конфигурация

Все эксперименты используют общие конфигурации из `shared/configs.py`:

- **Модели**: базовые, маленькие и большие конфигурации GPT
- **Токенизаторы**: параметры BPE обучения
- **Обучение**: гиперпараметры обучения
- **Генерация**: параметры генерации текста

## 📈 Результаты

Эксперименты сохраняют:
- Обученные модели в `checkpoints/`
- Токенизаторы в формате JSON
- Логи обучения и генерации
- Конфигурации моделей

## 🎯 Примеры использования

### Автономное использование (только llm)

```python
from llm.models.gpt import GPT
from llm.tokenizers import BPETokenizer

# Загрузка обученной модели
model = GPT(config)
model.load_state_dict(torch.load("checkpoints/gpt-bpe/model.pt"))

# Загрузка токенизатора
tokenizer = BPETokenizer.load("checkpoints/bpe_tokenizer.json")

# Генерация текста
input_ids = tokenizer.encode("промпт")
generated = model.generate(input_ids)
```

### Интеграция с HF (через hf-proxy)

```python
from hf_proxy import HFAdapter, HFTokenizerAdapter

# Загрузка через адаптеры
hf_model = HFAdapter.from_pretrained("checkpoints/hf-trained/pytorch_model.bin")
hf_tokenizer = HFTokenizerAdapter.from_pretrained("checkpoints/hf-bpe-tokenizer")

# Использование с HF инструментами
from transformers import pipeline
pipe = pipeline("text-generation", model=hf_model, tokenizer=hf_tokenizer)
```

## 🔍 Мониторинг

- **Логи обучения**: автоматически сохраняются в JSON
- **Метрики**: loss, длина генерации, эффективность токенизации
- **Визуализация**: можно интегрировать с TensorBoard через HF Trainer

## 🛠️ Разработка

### Добавление нового эксперимента

1. Создайте файл в соответствующей директории (`llm_only/` или `hf_integration/`)
2. Используйте общие утилиты из `shared/`
3. Сохраняйте результаты в стандартизированные пути
4. Документируйте конфигурации и результаты

### Модификация конфигураций

Измените соответствующие секции в `shared/configs.py`:
- `BASE_GPT_CONFIG` - параметры модели
- `BPE_CONFIG` - параметры токенизатора  
- `TRAINING_CONFIG` - параметры обучения
- `GENERATION_CONFIG` - параметры генерации

## 📚 Дополнительные ресурсы

- [Документация llm библиотеки](../llm/README.md)
- [Документация hf-proxy](../hf-proxy/README.md)
- [Примеры использования](../notebooks/)
