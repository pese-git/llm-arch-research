# Быстрый старт
<!-- description: Сквозной пример: взять текст, подготовить корпус, обучить небольшую LLaMA, прервать и продолжить обучение, сгенерировать текст из скрипта и из Python. -->

[← Установка](installation.md) · [Оглавление](README.md) · [Модели и конфиги →](models.md)

Здесь мы пройдём весь путь за несколько минут: возьмём три русских романа, подготовим из них корпус, обучим небольшую LLaMA (6 млн параметров), прервём и продолжим обучение и сгенерируем текст. Все команды запускаются **из корня репозитория** после [установки](installation.md); каждая выполнена, цифры ниже — из реального прогона на Apple Silicon (MPS).

Что получится, честно: модель научится русским словам, склонениям и характерным оборотам, но не связному смыслу — на 2.7 МБ текста и за 4 минуты иначе не бывает. Цель примера — показать, как устроен процесс, а качество растёт с корпусом и размером модели.

## 1. Корпус

Модели нужен один текстовый файл в UTF-8, в котором документы (здесь — романы) разделены пустой строкой. Возьмём три произведения из набора [RussianNovels](https://github.com/JoannaBy/RussianNovels) (другие источники — в разделе [«Где взять корпус»](data.md#где-взять-корпус)):

```bash
mkdir -p data/raw && cd data/raw
BASE=https://raw.githubusercontent.com/JoannaBy/RussianNovels/master/corpus
for f in Bulgakov_Master Bulgakov_BelayaGvardiya Dostoyevsky_BednyeLyudi; do
  curl -sLO "$BASE/$f.txt"
done
cd ../..

# склеить романы, поставив между ними пустую строку
for f in Bulgakov_Master Bulgakov_BelayaGvardiya Dostoyevsky_BednyeLyudi; do
  cat "data/raw/$f.txt"; printf '\n\n'
done > data/novels.txt
```

Теперь превратим текст в файлы токенов (подробнее — в разделе [«Корпус из файла»](data.md#корпус-из-файла)):

```bash
uv run python experiments/shared/prepare_corpus.py --input data/novels.txt --out data/novels \
  --vocab-size 4000 --tokenizer-lines 1500
```

Скрипт делит текст на обучающую и валидационную части, обучает BPE-токенизатор и записывает токены. Вывод:

```
✂️  Строк: train 13068, val 125
🔧 Обучение BPE на 1472 строках, vocab_size=4000...
✅ Токенизатор обучен за 55 с: data/novels/tokenizer.json (vocab_size=4004)
✅ data/novels/train.bin: 509673 токенов за 2 с
✅ data/novels/val.bin: 12684 токенов за 0 с
```

Символьный BPE на Python обучается медленно, поэтому `--tokenizer-lines` ограничивает число строк для словаря; сами файлы токенизируются по всему тексту.

## 2. Обучение

Обучение запускается скриптом [`run_llm_experiment.py`](../../experiments/llm_only/run_llm_experiment.py) с JSON-конфигом. Для этого примера есть готовый [`llama_quickstart_train.json`](../../experiments/llm_only/configs/llama_quickstart_train.json):

```bash
uv run python experiments/llm_only/run_llm_experiment.py \
  --model llama --action train --config experiments/llm_only/configs/llama_quickstart_train.json
```

Что в конфиге:

| Секция | Значение | Смысл |
|---|---|---|
| `data` | `train.bin`, `val.bin`, `tokenizer.json` | пути, которые записал `prepare_corpus.py` |
| `model_config` | 4 слоя, `embed_dim` 256, 4 головы, контекст 128 | модель LLaMA на 6 млн параметров; `vocab_size` подставляется из токенизатора |
| `training.max_steps` | 1000 | обучение по шагам, около 8 проходов по корпусу |
| `training.learning_rate`, `warmup_ratio` | 0.001, 0.05 | пиковый learning rate и доля шагов на warmup |
| `training.device` | `auto` | `cuda`, иначе `mps`, иначе `cpu`; `cpu` — принудительно на процессоре |
| `training.eval_interval`, `save_interval` | 250 | валидация и чекпоинт каждые 250 шагов |
| `training.seed` | 0 | порядок батчей воспроизводим, продолжение обучения точное |

Ключи `training` повторяют аргументы `Trainer`, их полный список — в [Обучении](training.md#trainer). Конфиг другой модели — `--model gpt`, `gpt2`, `mistral`, `mixtral`, `gemma` и свой `model_config` ([ключи по моделям](models.md)).

Обучение печатает валидационный loss каждые 250 шагов. В конце — перплексия на валидации:

```
Validation loss: 4.8563     # шаг 250
Validation loss: 4.2973     # шаг 500
Validation loss: 4.1318     # шаг 750
Validation loss: 4.0924     # шаг 1000
📈 val_perplexity: 59.8822
```

Перплексия 60 означает, что модель в среднем выбирает как бы из 60 равновероятных токенов; модель, которая ничего не знает, дала бы 4004 — размер словаря. Весь прогон занял около 4 минут на MPS; на CPU — заметно дольше.

Результаты в `checkpoints/llama-quickstart/`:

| Файл | Что внутри |
|---|---|
| `model.pt` | итоговая модель: класс, конфиг и веса (25 МБ) |
| `best.pt`, `last.pt` | чекпоинты обучения: те же веса плюс состояние оптимизатора и расписания (по 75 МБ); `best.pt` — шаг с лучшим валидационным loss |
| `config.json` | конфиг модели для чтения глазами |
| `log.json` | шаг, learning rate, train и val loss по каждой валидации |

### Прервать и продолжить

Остановите обучение в любой момент (`Ctrl+C`) и запустите ту же команду с `--resume`:

```bash
uv run python experiments/llm_only/run_llm_experiment.py \
  --model llama --action train --config experiments/llm_only/configs/llama_quickstart_train.json \
  --resume checkpoints/llama-quickstart/last.pt
```

```
⏯️  Продолжение с шага 250: checkpoints/llama-quickstart/last.pt
Validation loss: 4.2973
```

`last.pt` перезаписывается каждые `save_interval` шагов, поэтому продолжение начнётся с последнего такого шага (здесь 250), а не с момента остановки. Скрипт восстанавливает веса, моменты Adam, расписание, шаг и порядок батчей, поэтому кривая loss продолжается без скачка. Конфиг при этом должен быть тот же: `resume` отвергает чекпоинт, если `learning_rate`, `max_steps`, `batch_size` или warmup другие. Точное совпадение с непрерывным обучением гарантируется на CPU без dropout; на GPU — с точностью ядер. Подробнее — в разделе [«Чекпоинты и продолжение»](training.md#чекпоинты-и-продолжение).

## 3. Генерация

### Из скрипта

Тот же скрипт с действием `generate` и конфигом [`llama_quickstart_generate.json`](../../experiments/llm_only/configs/llama_quickstart_generate.json) дописывает каждый из `test_prompts`:

```bash
uv run python experiments/llm_only/run_llm_experiment.py \
  --model llama --action generate --config experiments/llm_only/configs/llama_quickstart_generate.json
```

```
[RESULT] Prompt: 'Мастер и Маргарита'
---
Мастер и Маргарита знала – остроено. Но мы, как это вы, как на месте, то на него, в которую, что она! ...
```

Параметры сэмплирования — секция `generation` конфига: `temperature`, `top_k`, `top_p`, `max_new_tokens`, `do_sample`. Веса задаёт `model_weights`: подойдёт и итоговый `model.pt`, и чекпоинт `best.pt` с лучшим валидационным loss.

### Из Python

```python
import torch
from llm.models.llama import Llama
from llm.tokenizers import BPETokenizer

tokenizer = BPETokenizer.load("data/novels/tokenizer.json")
model = Llama.load("checkpoints/llama-quickstart/best.pt")   # конфиг внутри файла, режим eval

prompt = torch.tensor([tokenizer.encode("Мастер и Маргарита")])

torch.manual_seed(0)
out = model.generate(prompt, max_new_tokens=60, do_sample=True, temperature=0.8, top_k=40)
print(tokenizer.decode(out[0].tolist()))

out = model.generate(prompt, max_new_tokens=40, do_sample=False)      # greedy
print(tokenizer.decode(out[0].tolist()))
```

`Llama.load` читает и `model.pt`, и чекпоинты обучения. Что делает каждый параметр `generate`, как остановить генерацию по `<eos>` и как подать несколько промптов — в разделе [Генерация](generation.md).

Greedy-генерация быстро зацикливается («что она не замечает, что она не замечает…»), а сэмплирование с `temperature` около 0.8 и `top_k` даёт более разнообразный, но и более шумный текст — это нормально для модели такого размера.

## То же без скрипта

Скрипт — тонкая обёртка над библиотекой. Обучение из Python, с теми же файлами из шага 1:

```python
from torch.utils.data import DataLoader

from llm.datasets.token_block_dataset import TokenBlockDataset
from llm.evaluation import perplexity
from llm.models.llama import Llama
from llm.tokenizers import BPETokenizer
from llm.training.trainer import Trainer

tokenizer = BPETokenizer.load("data/novels/tokenizer.json")
train = TokenBlockDataset("data/novels/train.bin", block_size=128)
val = TokenBlockDataset("data/novels/val.bin", block_size=128)

model = Llama({"vocab_size": tokenizer.get_vocab_size(), "embed_dim": 256, "num_heads": 4,
               "num_layers": 4, "max_position_embeddings": 128, "dropout": 0.1})
trainer = Trainer(model, train, val, lr=1e-3, batch_size=32, device="auto", max_steps=1000,
                  warmup_ratio=0.05, eval_interval=250, checkpoint_dir="checkpoints/py-run",
                  save_interval=250, seed=0)
trainer.train()                    # продолжить: trainer.resume("checkpoints/py-run/last.pt") до train()

print("perplexity:", perplexity(model, DataLoader(val, batch_size=32), device=trainer.device))
model.save("checkpoints/py-run/model.pt")
```

`block_size` датасетов равен `max_position_embeddings` модели: блок — это один обучающий пример. Дальше — генерация из Python, как выше.

## Что дальше

- **Свой корпус.** Подставьте любой текст в UTF-8 в `prepare_corpus.py --input` и поменяйте пути в секции `data`. На 2.7 МБ модель быстро переобучается: в прогоне с `max_steps` 1500 валидационный loss был минимальным на шаге 1000 (4.11) и вырос до 4.16 к шагу 1500. Поэтому в примере 1000 шагов, а `eval_interval` и `best.pt` нужны, чтобы не потерять лучший момент. Больше текста и чуть больше модель дают заметно лучший результат.
- **Другая архитектура.** Тот же корпус и скрипт, но `--model gpt`/`mistral`/… и свой `model_config`: так сравнивают архитектуры на одних данных по перплексии.
- **Устройство.** `device` в `training` — `cuda`, `mps`, `cpu` или `auto`. Видеокарта для примера не нужна.
- **Быстрая проверка.** Уменьшите `max_steps` до 40: перплексия после 40 шагов около 700, зато весь путь от обучения до генерации проходит быстро.
- **Как это устроено.** [Обучение](training.md) — параметры `Trainer`, расписание, чекпоинты; [Токенизатор и данные](data.md) — корпус и блоки; [глава «Обучение»](../textbook/training.md) учебника — почему AdamW, warmup и clipping.
