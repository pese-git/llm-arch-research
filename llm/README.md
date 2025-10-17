# LLM Framework - Фреймворк для языковых моделей

Модульная библиотека для создания, обучения и использования больших языковых моделей (LLM) с поддержкой различных архитектур (GPT, LLaMA и др.).

## 🏗️ Архитектура

Библиотека построена по модульному принципу с четким разделением ответственности:

```
llm/
├── core/                 # Базовые компоненты
│   ├── base_model.py    # Абстрактный базовый класс моделей
│   ├── cached_decoder.py # Универсальный декодер с кэшированием
│   ├── decoder.py       # Базовый декодер
│   ├── multi_head_attention.py # Многоголовое внимание
│   ├── head_attention.py # Одно-головое внимание
│   ├── feed_forward.py  # Стандартный FFN слой
│   ├── token_embeddings.py # Векторные представления токенов
│   ├── positional_embeddings.py # Абсолютные позиционные эмбеддинги
│   ├── rope.py          # Rotary Positional Embeddings (RoPE)
│   ├── rms_norm.py      # RMS Normalization
│   ├── swi_glu.py       # SwiGLU активация
│   ├── silu.py          # SiLU активация
│   └── gelu.py          # GELU активация
├── models/              # Конкретные реализации моделей
│   ├── gpt/            # GPT архитектуры
│   │   ├── gpt.py      # Базовая GPT
│   │   ├── gpt2.py     # GPT-2 реализация
│   │   └── __init__.py
│   ├── llama/          # LLaMA архитектура
│   │   ├── llama.py    # LLaMA реализация
│   │   └── __init__.py
│   └── mistral/        # Mistral архитектура
│       ├── mistral.py  # Mistral реализация
│       └── __init__.py
├── tokenizers/          # Токенизаторы
│   ├── base_tokenizer.py # Базовый интерфейс
│   └── bpe_tokenizer.py # BPE токенизатор
├── datasets/            # Работа с датасетами
│   ├── text_dataset.py    # Стандартный датасет
│   └── streaming_text_dataset.py # Стриминговый датасет
└── training/           # Утилиты обучения
    ├── trainer.py      # Тренировочный цикл
    ├── optimizer.py    # Оптимизаторы
    └── scheduler.py    # Планировщики обучения
```

## 🧩 Ключевые компоненты

### BaseModel (`core/base_model.py`)
**Абстрактный базовый класс** для всех языковых моделей с единым интерфейсом.

```python
class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Прямой проход модели."""
    
    @abstractmethod
    def generate(self, input_ids: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        """Генерация текста."""
```

### CachedDecoder (`core/cached_decoder.py`)
**Универсальный декодер** с поддержкой dependency injection и кэширования KV-памяти.

```python
CachedDecoder(
    feed_forward_layer=FeedForward(...),  # или SwiGLU
    norm_layer=nn.LayerNorm,              # или RMSNorm
    rope=RoPE(...),                       # опционально
    # ... другие параметры
)
```

### RoPE (`core/rope.py`)
**Rotary Positional Embeddings** - ротационные позиционные эмбеддинги.

**Математическая основа:**
```
θ_i = base^(-2i/d)
q'_m = q_m * cos(mθ_i) + rotate(q_m) * sin(mθ_i)
```

### RMSNorm (`core/rms_norm.py`)
**Root Mean Square Normalization** - упрощенная нормализация без среднего.

**Формула:**
```
RMSNorm(x) = (x / RMS(x)) * w
где RMS(x) = sqrt(mean(x²) + eps)
```

### SwiGLU (`core/swi_glu.py`)
**Swish-Gated Linear Unit** - современная активация с gating mechanism.

**Формула:**
```
SwiGLU(x) = Swish(xW_g + b_g) ⊙ (xW_u + b_u) * W_d + b_d
```

## 🚀 Примеры использования

### Создание классической GPT модели
```python
from llm.models.gpt import GPT

config = {
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "max_position_embeddings": 1024,
    "dropout": 0.1
}

model = GPT(config)
```

### Создание GPT2 модели
```python
from llm.models.gpt import GPT2

config = {
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "max_position_embeddings": 1024,
    "dropout": 0.1
}

model = GPT2(config)
```

### Создание LLaMA модели
```python
from llm.models.llama import Llama
from llm.core.swi_glu import SwiGLU
from llm.core.rms_norm import RMSNorm

config = {
    "vocab_size": 32000,
    "embed_dim": 4096,
    "num_heads": 32,
    "num_layers": 32,
    "max_position_embeddings": 2048,
    "dropout": 0.1
}

model = Llama(config)
```

### Генерация текста
```python
# Прямой проход
output = model(input_ids, attention_mask)

# Генерация текста
generated = model.generate(input_ids, max_length=100)
```

## 📊 Входные и выходные данные

### Входные данные:
- `input_ids`: `Tensor[int64]` формы `[batch_size, seq_len]` - индексы токенов
- `attention_mask`: `Tensor[bool]` формы `[batch_size, seq_len]` - маска внимания
- `cache`: `List[Tuple[Tensor, Tensor]]` - кэш ключей-значений для генерации

### Выходные данные:
- `logits`: `Tensor[float32]` формы `[batch_size, seq_len, vocab_size]` - вероятности токенов
- `cache`: `List[Tuple[Tensor, Tensor]]` - обновленный кэш (при использовании)

## 🏆 Поддерживаемые архитектуры

### GPT (Original) Особенности
- ✅ Многоголовое внимание
- ✅ Layer Normalization (после внимания и FFN)
- ✅ GELU активация
- ✅ Learned positional embeddings
- ✅ Базовая архитектура трансформер-декодера

### GPT-2 Особенности
- ✅ Layer Normalization (перед вниманием и FFN)
- ✅ GELU активация
- ✅ Learned positional embeddings
- ✅ Кэширование KV для быстрой генерации
- ✅ Улучшенная инициализация слоёв

### LLaMA Особенности
- ✅ Rotary Positional Embeddings (RoPE)
- ✅ RMS Normalization вместо LayerNorm
- ✅ SwiGLU активация вместо GELU
- ✅ Оптимизированная структура декодера
- ✅ Эффективное кэширование KV-памяти

### Mistral Особенности
- ✅ Sliding Window Attention (оконное внимание)
- ✅ Grouped Query Attention (GQA)
- ✅ RoPE
- ✅ RMSNorm
- ✅ Разделённая архитектура на блоки с эффективным управлением памятью
- ✅ Совместимость с HuggingFace через hf-proxy

## 🤝 Интеграция с HuggingFace и BPE

- Встроенная поддержка собственных BPE токенизаторов и экспериментальная поддержка токенизаторов через HuggingFace (см. hf-proxy).
- hf-proxy — экспериментальный модуль! Совместимость с будущими версиями Transformers не гарантируется; API может меняться.
- Допускается загрузка/конвертация моделей в формат HF для использования экосистемы Transformers.
- Для запуска моделей с токенизаторами HF используйте `hf-proxy` и соответствующие эксперименты из `experiments/hf_integration/`.

## 🧪 Тестирование

Запуск всех тестов:
```bash
cd llm
python -m pytest tests/ -v
```

**Статус тестов:** ✅ 101+ тест, охвачены все основные компоненты (ядро, ядро-токенизация, архитектуры, обучение)

## 📚 Научные концепции

### Трансформерная архитектура
Основана на механизме **внимания**, позволяющем модели взвешивать важность разных частей входной последовательности.

**Формула внимания:**
```
Attention(Q, K, V) = softmax(Q·Kᵀ/√d_k)·V
```

### RoPE (Rotary Positional Embeddings)
Инновационный метод кодирования позиционной информации через **вращение векторов** в комплексном пространстве.

**Преимущества:**
- Относительное позиционное кодирование
- Лучшая экстраполяция на длинные последовательности
- Сохранение нормы векторов

### RMSNorm vs LayerNorm
**RMSNorm** устраняет вычитание среднего, что делает его более стабильным и эффективным при обучении больших моделей.

### SwiGLU vs GELU
**SwiGLU** с gating mechanism показывает лучшую производительность благодаря способности выборочно передавать информацию.

## 🔧 Настройка и расширение

Библиотека разработана с учетом **расширяемости**. Для добавления новой архитектуры:

1. **Наследоваться** от `BaseModel`
2. **Реализовать** обязательные методы `forward()` и `generate()`
3. **Использовать** модульные компоненты из `core/`
4. **Добавить** конфигурацию модели

### Пример расширения:
```python
class NewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Использование готовых компонентов
        self.decoder = CachedDecoder(...)
        
    def forward(self, input_ids, attention_mask=None):
        # Реализация прямого прохода
        pass
```

## 📄 Лицензия

Проект распространяется под MIT License.
