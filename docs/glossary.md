# Глоссарий

[Оглавление](README.md) · [Обозначения](notation.md)

Термины пособия в алфавитном порядке: сначала английские, затем русские. Ссылка ведёт в главу, где термин разобран подробно.

## A–Z

**AdamW** — оптимизатор Adam с «отделённым» (decoupled) затуханием весов: веса уменьшаются на $`\eta\lambda\theta`$ отдельно от адаптивного шага. Стандарт для обучения трансформеров. → [training.md](training.md)

**ALiBi** — способ кодирования позиции штрафом к оценкам attention, линейным по расстоянию. В репозитории не реализован. → [positional-encoding.md](positional-encoding.md)

**Attention (внимание)** — операция, в которой каждая позиция собирает информацию с других позиций с весами $`\mathrm{softmax}(QK^\top/\sqrt{d_h})`$. → [attention.md](attention.md)

**`attention_mask`** — внешняя маска `[B, T]`: 1 — настоящий токен, 0 — паддинг. → [masks.md](masks.md)

**Autoregressive (авторегрессивная) модель** — модель, которая порождает последовательность по одному элементу, каждый раз опираясь на уже порождённые. → [language-modeling.md](language-modeling.md)

**Batch (батч)** — несколько примеров, обрабатываемых одновременно; ось $`B`$ тензоров.

**Bias (сдвиг)** — вектор $`\mathbf{b}`$ в линейном слое $`\mathbf{x}W + \mathbf{b}`$. В LLaMA, Mistral, Mixtral и Gemma его нет; в библиотеке — ключ `bias`.

**BPE (Byte Pair Encoding)** — алгоритм токенизации, который начинает с символов и многократно сливает самую частую пару соседних токенов. → [tokenization.md](tokenization.md)

**Causal-маска** — маска, запрещающая позиции $`i`$ смотреть на позиции $`j > i`$ (в будущее). → [masks.md](masks.md)

**Checkpoint (чекпоинт)** — файл с весами (и, в этой библиотеке, конфигом) обученной модели; `model.save` / `Model.load`.

**Context window (контекст)** — максимальное число токенов, которое модель обрабатывает за раз, $`T_{\max}`$ (`max_position_embeddings`).

**Cross-entropy (перекрёстная энтропия)** — функция потерь $`-\ln p_{\text{верный токен}}`$, усреднённая по позициям. → [language-modeling.md](language-modeling.md)

**Decoder-only трансформер** — трансформер только из блоков декодера с causal-маской; архитектура всех моделей пособия. → [language-modeling.md](language-modeling.md)

**Dropout** — регуляризация: при обучении случайно обнуляет элементы с вероятностью $`p`$ и масштабирует остальные на $`1/(1-p)`$. → [training.md](training.md)

**Embedding (эмбеддинг)** — обучаемый вектор, сопоставленный токену; строка матрицы $`E \in \mathbb{R}^{V \times d}`$. → [embeddings.md](embeddings.md)

**EOS / BOS / PAD / UNK** — специальные токены: конец и начало последовательности, заполнитель, неизвестный токен. → [tokenization.md](tokenization.md)

**Expert (эксперт)** — одна из $`E`$ параллельных FFN-сетей слоя MoE. → [mixture-of-experts.md](mixture-of-experts.md)

**Feed-forward сеть (FFN, MLP)** — двух- или трёхслойное преобразование каждой позиции независимо, вторая половина блока трансформера. → [feed-forward.md](feed-forward.md)

**Fine-tuning (дообучение)** — продолжение обучения предобученной модели на целевой задаче.

**GeGLU** — gated FFN с GELU в гейте; используется в Gemma. → [feed-forward.md](feed-forward.md)

**GELU** — функция активации $`x\,\Phi(x)`$, где $`\Phi`$ — функция распределения стандартного нормального закона; в GPT используется её tanh-аппроксимация. → [feed-forward.md](feed-forward.md)

**GQA (Grouped Query Attention)** — attention, в котором группа голов Q делит одну пару голов K/V; $`1 < G < H`$. → [attention.md](attention.md)

**Gradient clipping** — ограничение нормы градиента перед шагом оптимизатора. → [training.md](training.md)

**Greedy decoding (жадная генерация)** — выбор на каждом шаге самого вероятного токена. → [generation.md](generation.md)

**Head (голова)** — одна из $`H`$ параллельных копий attention в своём подпространстве размера $`d_h`$. → [attention.md](attention.md)

**KV-кэш** — сохранённые ключи и значения прошлых позиций, чтобы при генерации не пересчитывать их. → [attention.md](attention.md), [generation.md](generation.md)

**LayerNorm** — нормализация вектора: вычитание среднего, деление на стандартное отклонение, обучаемые масштаб и сдвиг. → [normalization.md](normalization.md)

**Learning rate (скорость обучения)** — множитель $`\eta`$ шага оптимизатора. → [training.md](training.md)

**Load-balancing loss** — вспомогательный loss, поощряющий равномерную загрузку экспертов MoE. → [mixture-of-experts.md](mixture-of-experts.md)

**Logits** — ненормированные оценки $`\mathbf{z} \in \mathbb{R}^{V}`$ на выходе модели; вероятности получаются softmax. → [embeddings.md](embeddings.md)

**LM head (выходная проекция)** — линейный слой из $`\mathbb{R}^{d}`$ в $`\mathbb{R}^{V}`$, дающий logits. → [embeddings.md](embeddings.md)

**MHA (Multi-Head Attention)** — attention с $`H`$ головами Q и столькими же головами K/V. → [attention.md](attention.md)

**MoE (Mixture-of-Experts)** — слой из нескольких экспертов, из которых роутер выбирает $`k`$ на каждый токен. → [mixture-of-experts.md](mixture-of-experts.md)

**MQA (Multi-Query Attention)** — attention с одной парой K/V на все головы Q; $`G = 1`$. → [attention.md](attention.md)

**Nucleus sampling** — см. top-p.

**Perplexity (перплексия)** — $`e^{\mathcal{L}}`$, где $`\mathcal{L}`$ — средняя cross-entropy; «эффективное число вариантов», между которыми выбирает модель. → [language-modeling.md](language-modeling.md)

**Post-LN / Pre-LN** — расположение нормализации: после сложения с residual (GPT-1) или перед подблоком (GPT-2 и далее). → [normalization.md](normalization.md)

**Prefill / decode** — две фазы генерации: обработка всего промпта сразу и затем порождение по одному токену. → [generation.md](generation.md)

**Residual-связь** — прибавление входа подблока к его выходу: $`\mathbf{x} + F(\mathbf{x})`$. → [normalization.md](normalization.md)

**RMSNorm** — нормализация делением на среднеквадратичное значение без вычитания среднего. → [normalization.md](normalization.md)

**RoPE (Rotary Position Embedding)** — кодирование позиции поворотом пар координат Q и K на угол, пропорциональный позиции. → [positional-encoding.md](positional-encoding.md)

**`rope_theta` (база RoPE)** — основание, задающее частоты поворота $`\theta_i = \text{base}^{-2i/d_h}`$. → [positional-encoding.md](positional-encoding.md)

**Rolling buffer cache (кольцевой кэш)** — KV-кэш фиксированного размера $`W`$ для скользящего окна: K и V позиции $`i`$ записываются в ячейку $`i \bmod W`$. В библиотеке вместо него кэш обрезается срезом до последних $`W`$ позиций. → [mistral.md](mistral.md#кэш-ограниченный-окном)

**Router (роутер)** — линейный слой MoE, оценивающий, каким экспертам отдать токен. → [mixture-of-experts.md](mixture-of-experts.md)

**SiLU / Swish** — функция активации $`x\,\sigma(x)`$. → [feed-forward.md](feed-forward.md)

**Sliding window attention (скользящее окно)** — attention, в котором токен видит только ближайшее прошлое: в библиотеке — себя и $`W`$ предыдущих позиций ($`W + 1`$ позиций, $`0 \le i - j \le W`$), в HuggingFace — $`W`$ позиций вместе с собой. → [masks.md](masks.md), [mistral.md](mistral.md#ширина-окна-w--1)

**Softmax** — функция, превращающая вектор чисел в распределение вероятностей. → [notation.md](notation.md)

**SwiGLU** — gated FFN с SiLU в гейте; используется в LLaMA, Mistral, Mixtral. → [feed-forward.md](feed-forward.md)

**Teacher forcing** — обучение, при котором на вход подаётся настоящий текст, а не предсказания модели. → [language-modeling.md](language-modeling.md)

**Temperature (температура)** — делитель logits перед softmax при генерации; меньше — увереннее, больше — разнообразнее. → [generation.md](generation.md)

**Token (токен)** — единица текста для модели: символ, часть слова или слово; целое число от 0 до $`V-1`$. → [tokenization.md](tokenization.md)

**Top-k sampling** — выбор следующего токена только среди $`k`$ самых вероятных. → [generation.md](generation.md)

**Top-p sampling** — выбор среди минимального набора самых вероятных токенов с суммарной вероятностью не меньше $`p`$. → [generation.md](generation.md)

**Warmup** — начальный период обучения, в котором learning rate линейно растёт от нуля. → [training.md](training.md)

**Weight decay (затухание весов)** — регуляризация, на каждом шаге немного уменьшающая веса. → [training.md](training.md)

**Weight tying (связывание весов)** — использование одной матрицы для эмбеддингов на входе и для выходной проекции. → [embeddings.md](embeddings.md)

## А–Я

**Авторегрессия** — см. Autoregressive.

**Батч** — см. Batch.

**Голова attention** — см. Head.

**Контекст** — см. Context window.

**Маска** — матрица, запрещающая части пар «запрос — ключ» в attention. → [masks.md](masks.md)

**Паддинг** — дополнение коротких последовательностей pad-токенами до общей длины; правый (в конце) и левый (в начале). → [masks.md](masks.md)

**Позиционное кодирование** — способ сообщить модели порядок токенов. → [positional-encoding.md](positional-encoding.md)

**Претокенизация** — предварительное разбиение текста на слова и знаки перед BPE. → [tokenization.md](tokenization.md)

**Скрытое состояние** — вектор $`\mathbf{h}_t \in \mathbb{R}^{d}`$, которым модель представляет позицию $`t`$ между слоями.

**Словарь** — множество всех токенов, размер $`V`$. → [tokenization.md](tokenization.md)

**Функция потерь (loss)** — число, которое обучение минимизирует; для языковой модели — cross-entropy.
