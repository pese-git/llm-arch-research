# Токенизатор и данные
<!-- description: Токенизатор BPETokenizer, датасеты из строк и корпус из файла: паддинг, метки, токенизация в .bin и блоки без паддинга. -->

[← Модели и конфиги](models.md) · [Оглавление](README.md) · [Обучение →](training.md)

## BPETokenizer

`llm.tokenizers.BPETokenizer` — символьный BPE: словарь начинается с символов корпуса, слияния учатся внутри слов (текст заранее разбивается на слова, как в GPT-2: пробел прикрепляется к началу следующего слова, пунктуация — отдельно). Новый текст кодируется применением выученных слияний по порядку ранга, как в GPT-2 и HuggingFace. Как это работает — в главе [Токенизация](../textbook/tokenization.md).

```python
from llm.tokenizers import BPETokenizer

texts = ["Нейронные сети учатся на данных.", "Трансформеры обрабатывают последовательности."]

tokenizer = BPETokenizer()
tokenizer.train(texts=texts, vocab_size=300, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

ids = tokenizer.encode("Нейронные сети")                        # список int
ids_with_bos_eos = tokenizer.encode("Нейронные сети", add_special_tokens=True)
text = tokenizer.decode(ids)                                     # "Нейронные сети"

tokenizer.save("bpe_tokenizer.json")
tokenizer = BPETokenizer.load("bpe_tokenizer.json")
print(tokenizer.get_vocab_size(), tokenizer.pad_token_id, tokenizer.eos_token_id)
```

Что важно знать:

- **`vocab_size` в `train`** — число обученных токенов без специальных; итоговый словарь — `tokenizer.get_vocab_size()`. **Его и передавайте в конфиг модели**, иначе id специальных токенов выйдут за пределы матрицы эмбеддингов.
- На маленьком корпусе обучение останавливается раньше `vocab_size`, когда каждое слово уже стало одним токеном.
- **Специальные токены** добавляются в конец словаря, поэтому `pad_token_id` — не 0. Внутри текста они не распознаются: строка `"<eos>"` кодируется как обычные символы; вставить специальный токен можно только по id.
- **Неизвестный символ** (которого не было в корпусе) кодируется как `<unk>`. Без `<unk>` в `special_tokens` — `ValueError` с перечнем таких символов. `decode` по умолчанию выбрасывает специальные токены, включая `<unk>`, как `skip_special_tokens` в HF; увидеть их — `decode(ids, skip_special_tokens=False)`.
- **Модель и токенизатор — пара.** Модель, обученная с одним словарём, бессмысленна с другим. Для чужих весов нужен их токенизатор (например, из `transformers`).

## Датасеты

Три датасета из `llm.datasets` принимают список строк, токенизатор и `block_size` и возвращают словари с тремя тензорами формы `[block_size]`:

| Ключ | Что внутри |
|---|---|
| `input_ids` | токены строки; длиннее `block_size` — обрезаются, короче — дополняются `pad_token_id` справа |
| `attention_mask` | 1 для настоящих токенов, 0 для паддинга |
| `labels` | копия `input_ids`, на паддинге `-100` — такие позиции не входят в loss |

| Класс | Когда токенизирует | Особенность |
|---|---|---|
| `TextDataset` | один раз, в конструкторе | — |
| `StreamingTextDataset` | при каждом обращении | строки по-прежнему лежат в памяти списком |
| `TextWithSpecialTokensDataset` | в конструкторе | `add_bos=True` / `add_eos=True` добавляют BOS и EOS; при обрезке для них оставляется место |

```python
from llm.datasets.text_dataset import TextDataset

dataset = TextDataset(texts, tokenizer, block_size=32)
item = dataset[0]
print(item["input_ids"].shape, item["attention_mask"].sum(), (item["labels"] == -100).sum())
```

- **Каждая строка — один пример.** Эти датасеты не склеивают тексты и не режут длинный текст на блоки: всё после `block_size` токенов теряется. Для корпуса из файла используйте `TokenBlockDataset` — см. [Корпус из файла](#корпус-из-файла).
- **Сдвиг меток делает не датасет, а `Trainer`**: метки — копия входа, логит позиции *t* сравнивается с меткой *t + 1*. Не сдвигайте метки сами, иначе модель будет учиться предсказывать токен через один.
- **Свой датасет** должен возвращать `input_ids` и `labels` (и, если есть паддинг, `attention_mask`), с `-100` на позициях, которые не должны входить в loss. Паддинг отмечайте по месту, а не сравнением с `pad_token_id`: pad может совпадать с настоящим токеном.

## Корпус из файла

Для настоящего обучения корпус токенизируется один раз в файл токенов, а датасет читает из него непрерывные блоки — так режут поток текста GPT-2 и nanoGPT: без паддинга, без повторной токенизации на каждой эпохе, без загрузки корпуса в память.

```python
from torch.utils.data import DataLoader
from llm.datasets.tokenize_corpus import tokenize_file
from llm.datasets.token_block_dataset import TokenBlockDataset

n = tokenize_file("corpus.txt", tokenizer, "data/corpus/train.bin", eos_token_id=tokenizer.eos_token_id)
dataset = TokenBlockDataset("data/corpus/train.bin", block_size=256)
batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
print(n, len(dataset), batch["input_ids"].shape)      # токенов, блоков, torch.Size([8, 256])
```

**`tokenize_file(text_path, tokenizer, out_path, eos_token_id=None, dtype=None)`** читает текст построчно (UTF-8, пустая строка — граница документа), кодирует каждую строку `tokenizer.encode(line, add_special_tokens=False)` и пишет плоский массив `uint16` (словарь до 65536) или `uint32` без заголовка, а рядом — `<out_path>.json` с `dtype`, числом токенов и документов. Если задан `eos_token_id`, он ставится после каждого документа. Подходит любой токенизатор с `encode` и `get_vocab_size`.

**`TokenBlockDataset(tokens, block_size)`** принимает путь к файлу (читается через `numpy.memmap`, `dtype` берётся из `.json`) или массив в памяти. Пример `i` — токены с `i·block_size` по `i·block_size + block_size`; остаток короче блока отбрасывается; `len` = `num_tokens // block_size`. Возвращает `input_ids` и `labels` (копия входа, сдвиг делает `Trainer`), `attention_mask` нет — паддинга в блоках не бывает. Блоки перемешивает `DataLoader(shuffle=True)`.

Скрипт `experiments/shared/prepare_corpus.py` делает всё сразу: делит текст на train/val, обучает или загружает BPE, пишет `train.bin`, `val.bin` и `tokenizer.json` — см. [experiments/README.md](../../experiments/README.md#корпус-из-файла). Пути к этим файлам указываются в секции `data` конфига обучения.

## Учебный корпус

В `experiments/shared/configs.py` лежит встроенный корпус `TRAIN_TEXTS` — 15 коротких русских предложений; `load_training_data()` из `experiments/shared/data.py` делит его 80/20. Он нужен, чтобы проверить, что обучение и генерация работают, а не чтобы получить качество: на 12 строках модель запоминает обучающие данные, валидационный loss остаётся около `ln V`. Для настоящих экспериментов подготовьте [корпус из файла](#корпус-из-файла).
