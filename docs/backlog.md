# Бэклог

Технический долг, найденный при разборе кода. Каждая запись: где проблема, как её воспроизвести, как исправить.

Приоритеты: **P1** — неверный результат или падение; **P2** — расхождение с документацией или статьёй, дешёвые исправления; **P3** — качество кода.

Бэклог составлен по `master` после влития `fix/kv-cache` (PR #8) и `feat/gpt-activation` (PR #9), на 2026-09-28; описания «Что» и «Воспроизведено» относятся к тому состоянию. Строки «Статус» обновляются по мере исправлений и отражают `master` после PR #48. Пункты с пометкой «воспроизведено» проверены запуском (torch 2.8). Величины расхождений при префилле кусками зависят от seed и приведены для порядка.

Пункты 49–55 добавлены при сверке бэклога с кодом и первоисточниками, пункт 56 — при исправлении пункта 3, пункты 57–62 — при написании учебного пособия (раздел «Токенизатор, данные и обучение»). Они стоят в разделах своих архитектур, номера не перенумерованы, чтобы не ломать перекрёстные ссылки.

## GPT-1

Модель: [`models/gpt/gpt.py`](../llm/src/llm/models/gpt/gpt.py), блок: [`core/gpt_decoder.py`](../llm/src/llm/core/gpt_decoder.py). Пункты 2, 3, 8 и 10 касаются общих модулей и затрагивают и другие архитектуры.

### Баги

#### 1. `generate` падает за пределами `max_position_embeddings` — P1

- **Где:** `GPT.generate`, `GPT.forward`, `PositionalEmbeddings.forward`.
- **Что:** контекст не обрезается. Проверки длины смотрят на `seq_len`, а не на `start_pos + seq_len`: в `GPT.forward` (`x.size(1)`), в `PositionalEmbeddings.forward` и в `MultiHeadAttention.forward` (проверяется только текущий кусок, без длины кэша). `GPT2.forward` при переданном кэше пропускает проверку совсем.
- **Воспроизведено:** `max_position_embeddings=16`, промпт 10 токенов, `max_new_tokens=10`. С кэшем — `IndexError: index out of range in self` из `nn.Embedding` (на CUDA — device-side assert). Без кэша — `ValueError`.
- **Исправление:** при `x.size(1) > max_seq_len` обрезать окно `x[:, -max_seq_len:]` и пересчитывать без кэша (абсолютные позиции сдвигаются, кэш становится невалидным). Проверять `start_pos + seq_len` в `forward`, `PositionalEmbeddings` и `MultiHeadAttention`.
- **Статус:** исправлено в ветке `fix/p1-bugs` для всех шести моделей: общие `next_generation_input` и `check_sequence_length` в `core/generation.py`, проверки `start_pos + seq_len` в `forward` моделей, `PositionalEmbeddings` и всех трёх модулях attention. Генерация с кэшем и без совпадает с эталоном, пересчитывающим последние `max_position_embeddings` токенов на каждом шаге (`test_kv_cache.py`).

#### 2. Нет causal-маски при кэше и `seq_len > 1` — P1

- **Где:** [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py), `if cache is None: scores = scores.masked_fill(...)`.
- **Что:** если передать кэш и несколько токенов (префилл кусками, спекулятивное декодирование), будущие токены внутри куска не маскируются. `generate` подаёт по одному токену, поэтому там не проявляется.
- **Воспроизведено:** префилл 4 + 6 токенов через кэш расходится с полным forward на 0.21 по логитам.
- **Исправление:** всегда накладывать маску со сдвигом `self._tril_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]`.
- **Статус:** исправлено в ветке `fix/p1-bugs` (вместе с пунктами 27 и 41): префилл любыми кусками совпадает с полным forward до 5e-7 во всех шести моделях (`test_kv_cache.py`).

#### 3. `attention_mask` молча игнорируется — P1

- **Где:** принимается в `GPT.forward`, `GptDecoder.forward`, `MultiHeadAttention.forward`, но нигде не применяется. У `GPT2.forward` такого параметра нет вовсе (передача даёт `TypeError`), маску принимает только `GPT2.generate`. `hf-proxy/src/hf_proxy/hf_adapter.py` в `forward` маску отбрасывает (`self.llm_model(input_ids)`), а в `generate` передаёт, ожидая, что она сработает.
- **Воспроизведено:** `attention_mask` из нулей даёт логиты, побитово равные вызову без маски.
- **Уточнение:** при правом паддинге маска не нужна: causal-маска и так не даёт настоящим токенам смотреть на стоящий после них паддинг (проверено: выход совпадает до 3e-7). Молча неверный результат получается при левом паддинге (расхождение 0.9–1.9) и при генерации после паддинга. Обучение через hf-proxy не страдало: коллатор дополняет справа.
- **Исправление:** пробросить маску до attention и накладывать её вместе с causal-маской (`[B, T]` → `[B, 1, 1, T_kv]`). Либо, пока не реализовано, убрать параметр или бросать `NotImplementedError`, чтобы не было молчаливой ошибки.
- **Статус:** исправлено в ветке `fix/p1-bugs` вторым способом. `forward` всех шести моделей принимает `attention_mask`; маска из единиц и правый паддинг допускаются, на остальные маски с нулями (левый паддинг, пропуски, нули с кэшем, любые нули в `generate`) — `NotImplementedError` (`check_attention_mask` в `core/generation.py`). `hf_adapter.forward` передаёт маску в модель. Поддержка левого паддинга вынесена в пункт 56; объяснение масок — в [masks.md](masks.md).

#### 4. `generate` не валидирует аргументы, хотя докстринг обещает — P2

- **Где:** `GPT.generate`.
- **Что:** в докстринге описаны `ValueError` при `temperature ≤ 0`, одновременных `top_k` и `top_p`, `top_k ≤ 0`, `top_p ∉ (0, 1]`. В коде проверок нет.
- **Воспроизведено:** `temperature=0.0` и `top_k=5, top_p=0.9` принимаются молча. При `temperature ≤ 0` (в том числе отрицательной) масштабирование просто пропускается. `top_k=0` падает с невнятным `RuntimeError: probability tensor contains either inf, nan or element < 0`.
- **Исправление:** добавить проверки из докстринга в начало метода.
- **Статус:** исправлено в ветке `test/tokenizer-and-temperature`: общая `validate_sampling_args` (`core/generation.py`) вызывается в начале `generate` всех шести моделей. Проверки действуют при `do_sample=True`; при жадной генерации параметры сэмплирования не влияют на результат и не проверяются (`temperature=0` допустима).

#### 49. Top-p отбрасывает токен, пересекающий порог — P2

- **Где:** `generate` во всех шести моделях, `sorted_mask = cum_probs <= top_p`.
- **Что:** маска оставляет только токены, у которых накопленная вероятность *включая сам токен* не больше `top_p`. Токен, на котором сумма переходит порог, выкидывается, хотя в nucleus sampling (Holtzman et al., 2019; `TopPLogitsWarper` в HF) он входит в ядро. При вероятностях `[0.5, 0.3, 0.2]` и `top_p=0.7` остаётся один токен вместо двух; при `top_p` меньше вероятности самого частого токена ядро держится только за счёт принудительного первого токена.
- **Исправление:** сдвинуть маску — `cum_probs - sorted_probs < top_p` (или `sorted_mask[..., 1:] = sorted_mask[..., :-1].clone(); sorted_mask[..., 0] = True`). Вносить в общую функцию выбора токена (пункт 18).
- **Попутно:** при `temperature ≤ 0` `logits_scaled` — тот же тензор, что `logits`, и запись `-inf` на месте в ветке top-p портит выход `forward`. Сейчас безвредно, но после вынесения в общую функцию лучше клонировать.
- **Статус:** исправлено в ветке `refactor/shared-generate`: выбор токена — общая `sample_next_token` (`core/generation.py`), токен остаётся в ядре, если сумма вероятностей более вероятных токенов меньше `top_p` (как `TopPLogitsWarper` в HF). Логиты не изменяются на месте. На тех же весах и seed результат top-p поменялся у трёх моделей из шести, остальные режимы совпадают с прежними побитово.

#### 56. Нет поддержки левого паддинга — P2

- **Где:** все шесть моделей, `check_attention_mask` в [`core/generation.py`](../llm/src/llm/core/generation.py).
- **Что:** генерация батчем промптов разной длины требует левого паддинга: строки должны кончаться в одной позиции. Для этого нужна маска ключей во всех трёх модулях attention и сдвиг позиций для каждой строки батча: иначе первый настоящий токен получает позицию, равную длине паддинга, а от позиции зависят `PositionalEmbeddings` и RoPE. До исправления такие маски отклонялись с `NotImplementedError` (пункт 3).
- **Исправление:** `position_ids = attention_mask.cumsum(-1) − 1` (как в HF) с передачей в `PositionalEmbeddings` и `RoPE` по строкам батча; маска ключей `[B, 1, 1, T_kv]` вместе с causal-маской; в `generate` — продление маски единицами для новых токенов и хранение её рядом с кэшем. Тест: логиты настоящих токенов с левым паддингом совпадают с прогоном без паддинга.
- **Статус:** исправлено в ветке `feat/left-padding`: `padding_from_attention_mask` ([`core/padding.py`](../llm/src/llm/core/padding.py)) строит по `attention_mask` маску ключей и позиции `cumsum − 1`, как `position_ids` в HF; модели передают их через декодеры в `MultiHeadAttention`, `GroupedQueryAttention`, `MultiQueryAttention` (параметр `padding`), `RoPE` и `PositionalEmbeddings` (параметр `positions`). Паддинг поддерживается в любом месте строки в `forward`, с кэшем маска всегда покрывает кэш (`[batch, cache_len + seq_len]`, даже из одних единиц — иначе паддинг в кэше остался бы незамаскированным без ошибки); `generate` принимает левый паддинг, наращивает маску и обрезает её вместе с окном `max_seq_len`, правый паддинг в `generate` — `ValueError`. pad-токен как запрос видит только себя (иначе NaN). Без маски или с маской из единиц — побитово прежний результат. Проверено для всех шести моделей: каждая строка батча — как без паддинга, в `forward`, с кэшем и в `generate`, в том числе за пределами `max_position_embeddings`; сверка с HF (GPT-2, LLaMA, Mistral с окном, Gemma) на левом паддинге: логиты настоящих токенов и greedy-генерация совпадают.

### Отклонения от GPT-1

#### 5. Нет weight tying — P2

- **Что:** в оригинальном коде OpenAI (`finetune-transformer-lm/train.py`: `tf.matmul(h, we, transpose_b=True)`, без bias) и в HuggingFace (`OpenAIGPTLMHeadModel.lm_head`, без bias, привязан к `tokens_embed`) выходная проекция делит веса с токенными эмбеддингами. В тексте статьи GPT-1 это явно не сказано — следует из кода. Здесь `_linear` — отдельный `nn.Linear` с bias.
- **Последствия:** примерно на `vocab_size × embed_dim` параметров больше; веса `openai-community/openai-gpt` напрямую не загружаются.
- **Исправление:** `_linear = nn.Linear(embed_dim, vocab_size, bias=False)` и `_linear.weight = _token_embeddings._embedding.weight`, под флагом конфига, если нужна обратная совместимость чекпойнтов.
- **Статус:** исправлено в ветке `feat/gpt-weight-tying`: ключ конфига `tie_word_embeddings` (по умолчанию `false` — прежняя отдельная проекция с bias, старые чекпоинты загружаются). С `true` `_linear` создаётся без bias и делит параметр с `_token_embeddings._embedding.weight` (`output_projection` в `core/token_embeddings.py`). Веса `openai-community/openai-gpt` загружаются через `convert_hf_state_dict` (`models/gpt/hf_weights.py`): число параметров совпадает с HF (116 534 784), логиты — до 2.3e-5, greedy-генерация на 20 токенов — токен в токен. Заодно выяснилось, что `afn="gelu"` у HF OpenAIGPT — tanh-аппроксимация, то есть наш `activation` по умолчанию.

#### 6. Нет dropout на весах внимания — P3

- **Что:** в GPT-1 (разд. 4.1 статьи: «residual, embedding, and attention dropouts» 0.1; `attn_pdrop=0.1` в HF) dropout применяется к весам после softmax. Здесь есть только dropout после выходной проекции.
- **Исправление:** `weights = self._attn_dropout(F.softmax(scores, dim=-1))`, отдельным параметром.
- **Статус:** исправлено в ветке `feat/attention-dropout`: `MultiHeadAttention` принимает `attention_dropout` и применяет его к весам после softmax; `GPT` читает `config["attention_dropout"]`. По умолчанию `0.0` — обучение с текущими конфигами побитово прежнее (`nn.Dropout(0)` не расходует генератор случайных чисел); значение из статьи, `0.1`, задаётся в конфиге явно. Остальные два dropout GPT-1 (эмбеддинги и выходы подблоков) по-прежнему задаются общим `dropout`.

#### 7. Нет инициализации весов из статьи — P3

- **Что:** в статье (разд. 4.1) и в `train.py` веса инициализируются N(0, 0.02). В репозитории используется инициализация PyTorch по умолчанию: std весов `Linear` ≈ 0.1, эмбеддингов ≈ 1.0.
- **Исправление:** метод `_init_weights` (Linear/Embedding — `normal_(0, 0.02)`, bias — нули) и вызов `self.apply(...)` в `__init__`.
- **Статус:** исправлено в ветке `feat/gpt-init`: `init_normal_` (`core/weight_init.py`) применяется в `GPT.__init__` — Linear и Embedding N(0, 0.02), bias нули; std задаётся ключом `initializer_range`. Начальный loss свежей модели — 6.96 при `ln V = 6.91` (было 7.07), разброс логитов 0.32 вместо 0.58. Чекпоинты и их выход не затрагиваются: загрузка перезаписывает инициализацию.

### Качество кода

#### 8. `use_cache=True` по умолчанию и нет `torch.no_grad()` в `generate` — P2

- **Что:** при обучении `forward` возвращает ненужные K/V каждого слоя. `generate` без `no_grad` у вызывающего строит autograd-граф на всю генерацию.
- **Воспроизведено:** в `eval()` логиты имеют `requires_grad=True`, кэш возвращается по умолчанию.
- **Исправление:** `use_cache=False` по умолчанию в `forward` (проверить `Trainer` и `hf_adapter`), декоратор `@torch.no_grad()` на `generate`.
- **Статус:** исправлено в ветке `refactor/shared-generate`: у `forward` всех моделей `use_cache=False` по умолчанию (`Trainer` и `hf_adapter` вызывают без кэша, `generate` передаёт `use_cache` явно), `BaseModel.generate` под `@torch.no_grad()`.

#### 9. Приведение dtype внутри `FeedForward.forward` — P3

- **Где:** [`core/feed_forward.py`](../llm/src/llm/core/feed_forward.py).
- **Что:** `_layer1`/`_layer2` переприсваиваются во время forward, если dtype входа отличается. Это скрывает ошибки dtype и рассинхронизирует состояние оптимизатора.
- **Исправление:** убрать, приводить модель снаружи (`model.to(dtype)`) или использовать `torch.autocast`.
- **Статус:** исправлено в ветке `fix/feedforward-dtype`: приведение убрано. Воспроизведено до исправления: один вызов с fp16 навсегда переводил веса в fp16 (после возврата к fp32 веса отличались от исходных до 1.2e-4), под `torch.autocast` веса `FeedForward` становились bf16 при fp32 у остальных слоёв.

#### 10. Интерфейс `BaseModel` не соответствует моделям — P3

- **Где:** [`core/base_model.py`](../llm/src/llm/core/base_model.py).
- **Что:** объявлены `forward(input_ids, attention_mask) -> Tensor` и `generate(input_ids, max_length)`; `GPT` возвращает `(logits, cache)` и принимает `max_new_tokens`, `do_sample` и т.д.
- **Исправление:** привести абстрактные сигнатуры к фактическим.
- **Статус:** исправлено в ветке `refactor/shared-generate`: `BaseModel` объявляет фактический `forward(x, use_cache=False, cache=None, attention_mask=None) -> (logits, cache)` и реализует общий `generate`. Порядок параметров `GPT.forward` исторически другой (`x, attention_mask, use_cache, cache`); `generate` и адаптер вызывают его по именам.

#### 11. Документация противоречит коду — P2

- Докстринг `GptDecoder` называет блок «pre-LN» и приводит pre-LN псевдокод; в коде post-LN.
- Пример в докстринге `GptDecoder` использует `Decoder(...)` и ожидает от `decoder(x)` тензор, а возвращается кортеж.
- Докстринг `GptDecoder.forward` называет аргумент `mask` (на деле `attention_mask`) и обещает тензор на выходе.
- В References класса `GPT` битая ссылка на статью: `research-covers/languageunsupervised/` (нет дефиса, правильно `language-unsupervised`).
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: докстринги `GptDecoder` описывают post-LN, пример использует `GptDecoder` и кортеж на выходе, у `forward` описаны фактические параметры; ссылка на статью GPT-1 исправлена. Неиспользуемые параметры `mask` удалены из `forward` всех модулей attention и декодеров (маска паддинга проверяется в `forward` модели, пункт 3); тесты, которые передавали маску и проверяли только форму, заменены проверками встроенной causal-маски.

#### 12. Мусор в коде — P3

- ~~Закомментированный старый `generate` в конце `gpt.py`~~ — удалён вместе с копиями `generate` (пункт 18).
- Неиспользуемые импорты: `Optional`, `Dict` в `gpt.py` (`math` в `feed_forward.py` удалён).
- ~~Мёртвые проверки `hasattr(torch, "bool")`~~ — удалены в ветке `refactor/remove-dead-bool-checks` (см. пункт 32).
- ~~Сравнения `do_sample == True`, `top_k != None`~~ — были только в копиях `generate` и ушли вместе с ними (пункт 18).
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: неиспользуемые `Optional`, `Dict` удалены.

## GPT-2

Модель: [`models/gpt/gpt2.py`](../llm/src/llm/models/gpt/gpt2.py), блок: [`core/gpt2_decoder.py`](../llm/src/llm/core/gpt2_decoder.py).

Общие с GPT-1 пункты касаются GPT-2 так же и здесь не повторяются:
- **1**, **2**, **3** — генерация за `max_position_embeddings`, маска при кэше и `attention_mask` (исправлены). До исправления: с кэшем `IndexError`, без кэша `ValueError`; префилл 4 + 6 расходился с полным forward на 0.12–0.16; `GPT2.forward` не принимал `attention_mask`, `generate` принимал и игнорировал.
- **56** — нет поддержки левого паддинга (исправлен).
- **4**, **49** — валидация аргументов `generate` и top-p (исправлены).
- **8** — `use_cache=True` по умолчанию и нет `no_grad` (исправлен).
- **9** — dtype в `FeedForward` (исправлен).
- **10** — интерфейс `BaseModel` (исправлен).

### Отклонения от GPT-2

#### 13. GELU: точная erf-версия вместо tanh-аппроксимации — P2

- **Что:** `Gpt2Decoder` и `GptDecoder` использовали `nn.GELU()`, то есть erf. Оригинальный код OpenAI (`gpt-2/src/model.py`, `finetune-transformer-lm/train.py`) и HF (`GPT2Config.activation_function="gelu_new"`; в `modeling_openai` `ACT_FNS["gelu"]` — это `gelu_new`) используют tanh-аппроксимацию. Опция `'gelu_exact'` в `FeedForward` на деле подключала tanh-аппроксимацию.
- **Воспроизведено:** при одинаковых весах логиты отличались от эталона с tanh-GELU на ~1e-4. С tanh-GELU расхождение 5e-7.
- **Статус:** исправлено в ветке `fix/gelu-tanh`: `'gelu_exact'` переименован в `'gelu_tanh'`, `Gpt2Decoder` использует `'gelu_tanh'`, у GPT-1 это значение по умолчанию для `config["activation"]` (`'gelu'` — точный erf-вариант — остаётся доступным).

#### 14. Нет weight tying, у lm-head есть bias — P2

- **Что:** в оригинале (`gpt-2/src/model.py`: `tf.matmul(h, wte, transpose_b=True)`) и в HF (`GPT2LMHeadModel.lm_head`, `bias=False`, `tie_word_embeddings=True`) выходная проекция делит веса с `wte`. Здесь `_linear` — отдельный `nn.Linear` с bias.
- **Воспроизведено:** `m._linear.bias is not None`, `m._linear.weight is not m._token_embeddings._embedding.weight`.
- **Последствия:** для конфигурации 124M лишних ~38M параметров (`50257 × 768`). Веса `openai-community/gpt2` напрямую не загружаются.
- **Исправление:** как в пункте 5 для GPT-1.
- **Статус:** исправлено в ветке `feat/gpt-weight-tying`, как пункт 5. Веса `openai-community/gpt2` загружаются через `convert_hf_state_dict`: 124 439 808 параметров, как в HF (без tying — 163 087 441), логиты — до 7.6e-5, greedy-генерация — токен в токен.

#### 15. Нет dropout на весах внимания — P3

- **Что:** в HF-реализации GPT-2 (`attn_pdrop=0.1`) dropout применяется к весам после softmax. В `gpt-2/src/model.py` dropout нет вовсе — это код только для инференса. Здесь только dropout после выходной проекции (`resid_pdrop`).
- **Исправление:** общее с пунктом 6, так как `MultiHeadAttention` общий.
- **Статус:** исправлено в ветке `feat/attention-dropout`, как пункт 6: `GPT2` читает `config["attention_dropout"]`, по умолчанию `0.0`, в HF — `0.1`.

#### 16. Нет инициализации весов из статьи — P3

- **Что:** статья GPT-2 (разд. 2.3) масштабирует веса residual-слоёв на `1/√N`, где N — число residual-слоёв; в HF (`GPT2PreTrainedModel._init_weights`) это `0.02 / √(2·num_layers)` для `c_proj` в attention и MLP, остальные веса — N(0, 0.02). Значение 0.02 в статье не указано, оно из кода: в `gpt-2/src/model.py` 0.02 для весов и `wte`, но 0.01 для `wpe`, а масштабирования residual-проекций в коде нет. В репозитории инициализация PyTorch по умолчанию.
- **Исправление:** как в пункте 7, плюс `normal_(0, 0.02 / math.sqrt(2 * num_layers))` для `MultiHeadAttention._layer` и `FeedForward._layer2`.
- **Статус:** исправлено в ветке `feat/gpt-init`: как пункт 7, плюс `scale_residual_projections_` — `MultiHeadAttention._layer` и `FeedForward._layer2` каждого блока N(0, 0.02 / √(2·num_layers)), как в HF. `wpe` — 0.02, как в HF (в коде OpenAI 0.01).

### Качество кода

#### 17. `_tril_mask` сохраняется в `state_dict` — P2

- **Где:** [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py), `register_buffer('_tril_mask', ...)`. Затрагивает все модели на `MultiHeadAttention`.
- **Что:** буфер `max_seq_len × max_seq_len` persistent: попадает в каждый чекпоинт по одному на слой и привязывает чекпоинт к `max_seq_len`.
- **Воспроизведено:** ключи `_decoders.{i}._heads._tril_mask` в `state_dict`. При `max_position_embeddings=1024` это 1 МБ на слой.
- **Исправление:** `register_buffer(..., persistent=False)`. Старые чекпоинты при этом загружаются только с `strict=False` или после удаления ключей.
- **Статус:** исправлено в ветке `fix/nonpersistent-buffers` (вместе с 25, 31, 47): `register_buffer(..., persistent=False)`. Старые чекпоинты с этими ключами по-прежнему загружаются и со `strict=True`: `_load_from_state_dict` модуля отбрасывает устаревшие ключи. Проверено: чекпоинт, сохранённый прежним кодом, даёт побитово те же логиты (`test_state_dict.py`).

#### 18. `generate` скопирован в шесть моделей — P2

- **Где:** `generate` в `gpt.py`, `gpt2.py`, `llama.py`, `mistral.py`, `mixtral.py`, `gemma.py`.
- **Что:** логика temperature/top-k/top-p/sampling одинакова, поэтому каждое исправление (пункты 1, 4, 19, `hasattr(torch, "bool")`) нужно вносить шесть раз.
- **Исправление:** вынести выбор следующего токена в общую функцию (например, `core/sampling.py`) или в `BaseModel.generate` поверх `forward(x, use_cache, cache)`.
- **Статус:** исправлено в ветке `refactor/shared-generate`: один `generate` в `BaseModel` поверх `self(x, use_cache, cache)`, копии в шести моделях удалены вместе с их `max_seq_len` (свойство тоже в `BaseModel`). Выбор токена — `sample_next_token`, проверки — `validate_sampling_args`, `check_attention_mask` (с пункта 56 — `check_generation_mask`), `next_generation_input` в `core/generation.py`.

#### 19. Пограничные случаи в `generate` — P3

- **Что:**
  - `top_k > vocab_size` падает в `torch.topk`;
  - нет остановки по `eos_token_id`, всегда генерируется ровно `max_new_tokens`.
- **Воспроизведено:** `top_k=100` при `vocab_size=50` — `RuntimeError: selected index k out of range`.
- **Исправление:** `top_k = min(top_k, vocab_size)`; параметр `eos_token_id` с остановкой, когда все последовательности батча его сгенерировали.
- **Статус:** исправлено в ветке `refactor/shared-generate`: `top_k` ограничивается размером словаря. Параметры `eos_token_id` и `pad_token_id`: строка, сгенерировавшая `eos_token_id`, дальше заполняется `pad_token_id` (по умолчанию тем же `eos_token_id`), генерация останавливается, когда закончены все строки — как в HF.

#### 20. `head_size` без проверки делимости — P3

- **Где:** `GPT.__init__` и `GPT2.__init__`, `head_size=config["embed_dim"] // config["num_heads"]`.
- **Что:** при `embed_dim % num_heads != 0` размер головы молча усекается, и внимание работает в пространстве меньше `embed_dim`.
- **Воспроизведено:** `embed_dim=30, num_heads=4` принимается, `head_size=7`, Q/K/V — 28 измерений.
- **Исправление:** `assert`/`ValueError` в `__init__`. Связано с тем, что ключ `head_size` в конфигах не читается (см. [известные ограничения](README.md#известные-ограничения)).
- **Статус:** исправлено в ветке `fix/config-validation` (вместе с 28, 29, 36): общая `resolve_head_size` (`core/config_checks.py`) в конструкторе всех шести моделей. Без явного `head_size` неделимый `embed_dim` даёт `ValueError` с объяснением; в моделях с RoPE нечётный `head_size` — тоже `ValueError` с упоминанием `embed_dim` и числа голов (в `RoPE.__init__` `assert` заменён на `ValueError`).

#### 21. Документация и мусор — P3

- ~~Пример в докстринге модуля: `model.generate(input_ids, max_length=30)`~~ — исправлен на `max_new_tokens`/`do_sample` вместе с пунктом 18. В докстринге класса `model(input_ids)` по-прежнему описан как возвращающий логиты, а возвращается кортеж.
- Неиспользуемые импорты в `gpt2.py`: `FeedForward`, `Tensor`.
- Параметр `rope` в `Gpt2Decoder` (и импорт `RoPE`) — GPT-2 его не использует.
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: примеры в докстрингах `GPT2` распаковывают кортеж, неиспользуемые импорты удалены, параметр `rope` и импорт `RoPE` в `Gpt2Decoder` удалены.

## LLaMA

Модель: [`models/llama/llama.py`](../llm/src/llm/models/llama/llama.py), блок: [`core/cached_decoder.py`](../llm/src/llm/core/cached_decoder.py), attention: [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py) с [`core/rope.py`](../llm/src/llm/core/rope.py).

Общие с GPT пункты касаются LLaMA так же и здесь не повторяются:
- **1**, **2**, **3** — генерация за `max_position_embeddings`, маска при кэше и `attention_mask` (исправлены). До исправления: с кэшем падал `RuntimeError: shape '[1, 1, 1, <head_size / 2>]' is invalid for input of size 0` из `RoPE.forward` (пустой срез cos/sin), без кэша `ValueError`; префилл 4 + 6 расходился с полным forward на 0.14–0.28; `Llama.forward` не принимал `attention_mask`, `generate` его игнорировал. В моделях с RoPE контекст нельзя просто обрезать окном и продолжить с кэшем: при обрезке K пересчитываются заново с новыми позициями.
- **56** — нет поддержки левого паддинга (исправлен).
- **4**, **49** — валидация аргументов `generate` и top-p (исправлены).
- **8** — `use_cache=True` по умолчанию и нет `no_grad` (исправлен).
- **10** — интерфейс `BaseModel` (исправлен).
- **17** — `_tril_mask` в `state_dict` (исправлен).
- **18** — дублирование `generate` (исправлен).
- **19** — пограничные случаи `generate`: `top_k` больше словаря, остановка по `eos_token_id` (исправлен).
- **20** — проверка делимости `embed_dim` на число голов (исправлен).

### Баги

#### 22. `generate` молча принимает любые именованные аргументы — P2

- **Где:** `Llama.generate(..., attention_mask=None, **kwargs)`.
- **Что:** `**kwargs` нигде не используется, поэтому опечатки и аргументы из других API (`max_length`, `eos_token_id`) проглатываются без ошибки.
- **Воспроизведено:** `generate(x, 2, do_sample=False, max_lenght=5)` выполняется без ошибок.
- **Исправление:** `hf-proxy/src/hf_proxy/hf_adapter.py` пробрасывает `**kwargs` в `model.generate`, поэтому просто убрать параметр нельзя. Явно перечислить поддерживаемые ключи и бросать `TypeError` на остальных (или фильтровать ключи в адаптере).
- **Статус:** исправлено в ветке `refactor/shared-generate`: у общего `generate` нет `**kwargs`, неизвестный именованный аргумент — `TypeError`. `hf_adapter.generate` передаёт свои `**kwargs` как есть, поэтому опечатки через адаптер тоже дают `TypeError`; настоящий `transformers.pipeline` лишних ключей не передаёт (проверено).

### Отклонения от LLaMA

Докстринг `Llama` и [llama.md](llama.md#отличия-от-llama) уже упоминают bias и dropout; ниже — что из этого следует и чего там нет.

#### 23. SwiGLU с hidden = 4·d вместо ⅔·4·d — P2

- **Где:** [`core/swi_glu.py`](../llm/src/llm/core/swi_glu.py), `nn.Linear(emb_size, 4 * emb_size)` для `_gate`, `_up`, `_down`.
- **Что:** в LLaMA (разд. 2.2 статьи; `FeedForward` в `facebookresearch/llama/model.py`) скрытая размерность — `2/3 · 4d`, округлённая вверх до кратного `multiple_of=256`, чтобы три матрицы SwiGLU весили столько же, сколько две матрицы обычного FFN с `4d`. Здесь три матрицы по `4d`.
- **Последствия:** FFN примерно в 1.5 раза тяжелее, чем в статье. Для `d=4096`: hidden 16384 вместо 11008, ~201M вместо ~135M параметров FFN на слой. При сравнении с GPT той же ширины LLaMA получает лишние параметры, и сравнение архитектур становится нечестным.
- **Исправление:** параметр `hidden_dim` в `SwiGLU` (по умолчанию — формула LLaMA, опционально `multiple_of`). Затрагивает Mistral и Mixtral, которые используют тот же `SwiGLU`; меняет размеры весов, поэтому старые чекпоинты не загрузятся.
- **Статус:** исправлено для LLaMA в ветке `feat/llama-hf-parity`: `SwiGLU` принимает `hidden_dim`, `Llama` читает ключ `intermediate_size` (по умолчанию прежние `4 · embed_dim`, старые чекпоинты загружаются). Формула LLaMA — `llama_intermediate_size(embed_dim, multiple_of=256, ffn_dim_multiplier=None)`: 11008 для 7B, 13824 для 13B, 28672 для LLaMA 2 70B. Проверено на весах `nickypro/tinyllama-15M/42M/110M` (FFN 768, 1376, 2048 — по той же формуле с `multiple_of=32`): логиты совпадают с HF до 4e-5, greedy — токен в токен. Для Mistral и Mixtral — в пункте 30.

#### 24. Bias во всех `Linear` — P3

- **Что:** в LLaMA все проекции (`wq`, `wk`, `wv`, `wo`, `w1`–`w3`, `output`) без bias. Здесь bias есть в Q/K/V, выходной проекции attention, трёх матрицах SwiGLU и голове на словарь.
- **Воспроизведено:** `m._decoders[0]._heads._q.bias is not None`, `m._linear.bias is not None`.
- **Последствия:** веса Meta/HF LLaMA напрямую не загружаются (лишние ключи `*.bias`). Для загрузки весов HF, помимо bias, нужна перестановка строк `q_proj`/`k_proj`: HF использует `rotate_half` (половины вектора), а здесь, как у Meta, — чередующиеся пары `(2i, 2i+1)`.
- **Исправление:** флаг `bias` в конфиге (по умолчанию `False` для LLaMA) с пробросом в `MultiHeadAttention` и `SwiGLU`.
- **Статус:** исправлено для LLaMA в ветке `feat/llama-hf-parity`: ключ `bias` (по умолчанию `true`, прежнее поведение) пробрасывается в `MultiHeadAttention`, `CachedDecoder`, `SwiGLU` и голову. Веса HF загружаются через `convert_hf_state_dict` (`models/llama/hf_weights.py`) с перестановкой строк `q_proj`/`k_proj` под RoPE на чередующихся парах; сверено с пятью моделями (`nickypro/tinyllama-*`, `JackFram/llama-68m/160m`): логиты до 1.1e-4, greedy с KV-кэшем совпадает. Для Mistral и Mixtral — в ветке `feat/mistral-mixtral-hf-parity` (см. пункт 30).

### Качество кода

#### 25. RoPE-буферы в `state_dict`, по копии на каждый слой — P2

- **Где:** [`core/rope.py`](../llm/src/llm/core/rope.py), `register_buffer("cos_matrix", ...)` и `register_buffer("sin_matrix", ...)`. Один объект `RoPE` зарегистрирован в модели и в `MultiHeadAttention` каждого слоя.
- **Что:** буферы persistent, и `state_dict` содержит их под `num_layers + 1` ключами: `_position_embeddings.cos_matrix`, `_decoders.{i}._heads._rope.cos_matrix` и т.д. Чекпоинт хранит одни и те же таблицы многократно и привязан к `max_position_embeddings` — увеличить контекст без правки `state_dict` нельзя.
- **Воспроизведено:** при `num_layers=2` — три ключа `*.cos_matrix` и три `*.sin_matrix`.
- **Исправление:** `persistent=False` (как в пункте 17). Затрагивает все модели с RoPE: Mistral, Mixtral, Gemma.
- **Статус:** исправлено в ветке `fix/nonpersistent-buffers`: `cos_matrix` и `sin_matrix` с `persistent=False`, в `state_dict` их больше нет. Чекпоинт моделей с RoPE теперь загружается и в модель с большим `max_position_embeddings`. Старые чекпоинты с этими ключами по-прежнему загружаются и со `strict=True`: `_load_from_state_dict` модуля отбрасывает устаревшие ключи. Проверено: чекпоинт, сохранённый прежним кодом, даёт побитово те же логиты (`test_state_dict.py`).

#### 26. Документация и мусор — P3

- Закомментированный блок вычисления `start_pos` и строка `# pos_out = ...` в `Llama.forward`; ~~неиспользуемая переменная `vocab_size` в `generate`~~ (ушла вместе с копиями `generate`, пункт 18).
- Неиспользуемые импорты: `Tensor` в `llama.py`, `FeedForward` в `cached_decoder.py`, `Optional` в `rope.py`, `swi_glu.py`, `rms_norm.py`.
- Докстринг `CachedDecoder` описывает LayerNorm и GELU, хотя для LLaMA блок собирается с `RMSNorm` и `SwiGLU`.
- Комментарий к форме выхода в `RoPE.forward` — `[batch_size, seq_len, head_size]`, фактически 4D `[batch, num_heads, seq_len, head_size]`.
- В [README.md](README.md) устарели пометки «⚠️ без GQA, вопреки докстрингу» в таблице и пункт «LLaMA — нет GQA, вопреки докстрингу» в известных ограничениях: докстринг уже исправлен, расхождения больше нет.
- ~~Нет `save`/`load` ни в `Llama`, ни в `BaseModel`~~ — есть в `BaseModel` (пункт 35).
- [`tests/models/test_llama.py`](../llm/tests/models/test_llama.py) проверяет только формы. Кэшированная генерация по одному токену сверяется с полным forward в [`test_kv_cache.py`](../llm/tests/models/test_kv_cache.py) для всех моделей, там же — префилл кусками и генерация за `max_position_embeddings` (пункты 1, 2). Какие токены оставляют top-k/top-p, проверяет `test_generation.py` (пункт 49).
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: закомментированный код в `Llama.forward` и неиспользуемые импорты удалены, докстринг `CachedDecoder` описывает подставляемые нормализацию и FFN (для LLaMA — RMSNorm и SwiGLU), комментарий к форме выхода `RoPE` исправлен, устаревшие пометки о GQA в [README.md](README.md) убраны.

## Mistral

Модель: [`models/mistral/mistral.py`](../llm/src/llm/models/mistral/mistral.py), блок: [`core/mistral_decoder.py`](../llm/src/llm/core/mistral_decoder.py), attention: [`core/group_query_attention.py`](../llm/src/llm/core/group_query_attention.py) с [`core/rope.py`](../llm/src/llm/core/rope.py). `GroupedQueryAttention` общий с Mixtral, поэтому пункты 27–30 затрагивают и её.

Общие с предыдущими моделями пункты касаются Mistral так же и здесь не повторяются:
- **1**, **3** — генерация за `max_position_embeddings` и `attention_mask` (исправлены). До исправления с кэшем падал `RuntimeError: shape '[1, 1, 1, 4]' is invalid for input of size 0` из `RoPE.forward`. Для Mistral это было особенно заметно: sliding window и rolling-buffer кэш позволяют генерировать сколь угодно долго, мешала только таблица cos/sin. Теперь `generate` продолжает по последним `max_position_embeddings` токенам.
- **56** — нет поддержки левого паддинга (исправлен).
- **4**, **49** — валидация аргументов `generate` и top-p (исправлены).
- **8** — `use_cache=True` по умолчанию и нет `no_grad` (исправлен).
- **10** — интерфейс `BaseModel` (исправлен).
- **18** — дублирование `generate` (исправлен).
- **19** — пограничные случаи `generate`: `top_k` больше словаря, остановка по `eos_token_id` (исправлен).
- **22** — `**kwargs` в `generate` (исправлен).
- **24** — bias во всех `Linear`: у Mistral 7B проекции тоже без bias. До исправления: `_heads._q.bias is not None`, `_linear.bias is not None`. Исправлен ключом `bias` (пункт 30).
- **25** — RoPE-буферы в `state_dict` по копии на слой (исправлен).

### Баги

#### 27. Нет маски при кэше и `seq_len > 1` в `GroupedQueryAttention` — P1

- **Где:** `GroupedQueryAttention.forward`, `if cache is None: scores = scores.masked_fill(...)`.
- **Что:** то же, что пункт 2, но в отдельном модуле GQA, и ломается не только causal-часть, но и окно: токены куска видят будущее внутри куска, а ключи из кэша не обрезаются по окну для каждой строки. При одном новом токене маска не нужна: кэш содержит ровно `window_size` позиций, плюс сам токен — это `W + 1`, как и в маске без кэша.
- **Воспроизведено:** префилл 4 + 6 токенов через кэш расходится с полным forward на 0.2–0.3 по логитам (так же 5 + 5 и 6 + 4) при `window_size=4`. Генерация по одному токену с кэшем совпадает с полным forward (3.6e-7).
- **Исправление:** при кэше строить маску по абсолютным позициям: строки `start_pos … start_pos + T − 1`, столбцы — позиции ключей `start_pos − len(k_cache) … start_pos + T − 1`, разрешено `0 ≤ i − j ≤ window_size`.
- **Статус:** исправлено в ветке `fix/p1-bugs`: маска берётся срезом `_tril_mask[start_pos:start_pos + T, start_pos − cache_len:start_pos + T]` по абсолютным позициям. Префилл кусками, в том числе длиннее окна, совпадает с полным forward (`test_kv_cache.py`).

#### 28. Ключ `head_size` в конфиге игнорируется — P2

- **Где:** `Mistral.__init__`, `head_size=config["embed_dim"] // config["num_q_heads"]` для `RoPE` и `MistralDecoder`.
- **Что:** в [`mistral_train.json`](../experiments/llm_only/configs/mistral_train.json) задан `"head_size": 64`, и он совпадает с `256 // 4` случайно. Если изменить одно из значений, второе молча не подстроится.
- **Воспроизведено:** конфиг с `"head_size": 16` при `embed_dim=32, num_q_heads=4` даёт `head_size=8`.
- **Исправление:** читать `config.get("head_size", embed_dim // num_q_heads)` и передавать это значение и в `RoPE`, и в `MistralDecoder`. Если размер задан явно, `num_q_heads * head_size` может не равняться `embed_dim` — выходная проекция `_layer` это уже поддерживает.
- **Статус:** исправлено в ветке `fix/config-validation`: все шесть моделей читают `config.get("head_size")` и передают его и в attention, и в `RoPE`; без ключа — `embed_dim // <число голов>`. Если `head_size` задан, `num_heads · head_size` может отличаться от `embed_dim`.

#### 29. Нет проверок `num_q_heads` и `num_kv_heads` — P2

- **Где:** `GroupedQueryAttention.__init__`, `Mistral.__init__`.
- **Что:**
  - `num_q_heads % num_kv_heads != 0` принимается конструктором и падает только в первом `forward` внутри `_repeat_kv_heads` с непонятной ошибкой `reshape`;
  - `embed_dim % num_q_heads != 0` молча усекает размер голов (как пункт 20).
- **Воспроизведено:** `num_q_heads=4, num_kv_heads=3` — `RuntimeError: shape '[1, 4, 10, 8]' is invalid for input of size 240` при `forward`. `embed_dim=32, num_q_heads=3` — Q-проекция на 30 измерений.
- **Исправление:** `ValueError` в `__init__` с понятным сообщением для обоих условий.
- **Статус:** исправлено в ветке `fix/config-validation`: `GroupedQueryAttention.__init__` отклоняет `num_kv_heads < 1` и `num_q_heads`, не делящееся на `num_kv_heads`; неделимый `embed_dim` отклоняет `resolve_head_size` (пункт 20).

### Отклонения от Mistral 7B

#### 30. Размер скрытого слоя SwiGLU — P3

- **Что:** Mistral 7B использует `hidden_dim = 14336` при `dim = 4096` (3.5·d, `intermediate_size` в HF). Здесь `4·d` в каждой из трёх матриц, то есть FFN примерно на 14% тяжелее. Исправление общее с пунктом 23: параметр `hidden_dim` в `SwiGLU`, для Mistral — из конфига.
- **Статус:** исправлено в ветке `feat/mistral-mixtral-hf-parity`: `Mistral` и `Mixtral` читают ключ `intermediate_size` (по умолчанию прежние `4 · embed_dim`, старые чекпоинты загружаются); у Mixtral он задаёт размер каждого эксперта (`MoE(hidden_dim=...)`). Вместе с ним ключ `bias` (по умолчанию `true`) убирает bias из Q/K/V, выхода attention (`GroupedQueryAttention(bias=...)`), SwiGLU, роутера и головы — это закрывает остаток пункта 24 для Mistral и Mixtral. Веса `MistralForCausalLM`/`MixtralForCausalLM` загружаются через `convert_hf_state_dict` (общий с LLaMA, K переставляется по `num_key_value_heads`); сверено со случайными моделями HF: логиты до 1e-5, greedy с KV-кэшем дольше окна совпадает.

#### 50. `eps` в RMSNorm зашит как 1e-6 — P3

- **Где:** [`core/rms_norm.py`](../llm/src/llm/core/rms_norm.py), `RMSNorm(dim, eps=1e-6)`; все модели и декодеры создают `RMSNorm` без `eps`.
- **Что:** у Mistral 7B `norm_eps = 1e-5` (`rms_norm_eps` в HF), у LLaMA-1 — 1e-6, у Gemma — 1e-6. Задать значение из конфига нельзя. База RoPE 10 000 для LLaMA-1 и Mistral 7B v0.1 совпадает с оригиналом.
- **Исправление:** читать `config.get("rms_norm_eps", 1e-6)` и пробрасывать в `RMSNorm` модели и декодеров.
- **Статус:** исправлено в ветке `feat/rms-norm-eps`: LLaMA, Mistral, Mixtral и Gemma читают `config["rms_norm_eps"]` (по умолчанию `1e-6`) и передают его во все RMSNorm — по две в каждом блоке и финальную (LLaMA — через `functools.partial(RMSNorm, eps=...)` в `CachedDecoder`, декодеры Mistral, Mixtral и Gemma — параметром `norm_eps`). `eps ≤ 0` — `ValueError`. По умолчанию выход побитово прежний; `eps` не параметр, поэтому формат чекпоинтов не меняется. Конфиги экспериментов не менялись: для Mistral 7B и Mixtral 8x7B оригинальное значение `1e-5` задаётся в конфиге явно.

#### 51. Dropout в attention и FFN — P3

- **Что:** в Mistral 7B dropout нет (в `mistral-inference` его нет вовсе, в HF `attention_dropout=0.0`). Здесь dropout есть в `GroupedQueryAttention` и в `SwiGLU`. Для LLaMA это указано в докстринге и [llama.md](llama.md#отличия-от-llama), для Mistral — нигде.
- **Исправление:** задокументировать в [mistral.md](mistral.md) или ставить `dropout=0.0` по умолчанию.
- **Статус:** сделано в ветке `docs/mistral-gemma-dropout`: задокументировано в [mistral.md](mistral.md#отличия-от-mistral-7b) (новый раздел «Отличия от Mistral 7B») и в таблице конфигурации. Значение по умолчанию поменять нельзя: `dropout` — обязательный ключ конфига. Проверено, что `dropout: 0` обнуляет все пять dropout модели (после эмбеддингов и в attention и SwiGLU каждого блока), так что для соответствия оригиналу достаточно конфига. Dropout в attention у Mistral не на весах внимания, а на выходе — как и был.


### Качество кода

#### 31. `_tril_mask` в `state_dict` — P2

- **Где:** `GroupedQueryAttention.__init__`, `register_buffer("_tril_mask", ...)`.
- **Что:** то же, что пункт 17, но в `GroupedQueryAttention`, поэтому исправление в `MultiHeadAttention` его не закроет. Маска `max_seq_len × max_seq_len` хранится в каждом слое и привязывает чекпоинт к `max_seq_len` и `window_size`.
- **Воспроизведено:** ключи `_decoders.{i}._heads._tril_mask` в `state_dict`.
- **Исправление:** `persistent=False`, либо строить маску на лету по позициям (заодно закрывает пункт 27).
- **Статус:** исправлено в ветке `fix/nonpersistent-buffers` (как пункт 17).

#### 32. Совместимость с PyTorch < 1.2 сделана наполовину — P3

- **Где:** `GroupedQueryAttention.__init__` (`mask.bool() if hasattr(torch, "bool") else mask.byte()`), `~self._tril_mask[...]` в `forward`, top-k/top-p в `Mistral.generate`, `assert x.ndim == 4` в `RoPE.forward`.
- **Что:** на torch ≥ 1.2 `hasattr(torch, "bool")` всегда истинно, и uint8-ветка никогда не выполняется, то есть не тестируется. На torch < 1.2 она, скорее всего, логически верна: там `~` над `ByteTensor` было логическим НЕ (побитовым стало в 1.2, PyTorch PR #22326), а uint8-маски допустимы в `masked_fill` и индексации. На современном torch та же ветка сломалась бы (`~` для uint8 даёт `[254, 255, …]`), но выполниться там не может. Вероятнее ломает старый стенд другое: атрибута `Tensor.ndim` в torch 1.1, по всей видимости, ещё нет (не проверено запуском). Внешний стенд с torch < 1.2 прошла только версия на float-масках с `== 0`.
- **Исправление:** выбрать одно. Либо перейти на float-маски и `masked_fill(mask == 0, ...)` во всём коде и заменить `x.ndim` на `x.dim()`, либо отказаться от поддержки torch < 1.2 и убрать все `hasattr(torch, "bool")` (см. пункт 12).
- **Статус:** выбран второй вариант — поддержка torch < 1.2 в коде библиотеки не заявляется (`pyproject.toml` требует `torch>=2.3`), все 36 проверок `hasattr(torch, "bool")` удалены в ветке `refactor/remove-dead-bool-checks`. Выходы `forward`/`generate` всех шести моделей побитово совпадают с прежними. Код для стенда со старым torch переносится отдельно и использует float-маски.

#### 33. Документация и мусор — P3

- Докстринг `Mistral`: название статьи выдумано («Mistral: Fast and Efficient Dense and Mixture of Experts Transformer Models»), настоящее — «Mistral 7B».
- Докстринг `GroupedQueryAttention`: ссылка «Self-attention with linear complexity (Vila et al.) arXiv:2302.05442» не соответствует статье (arXiv:2302.05442 — «Scaling Vision Transformers to 22 Billion Parameters», Dehghani et al.); утверждение, что GQA используется в GPT-4, не подтверждено; обещано требование `num_q_heads * head_size == emb_size`, которое не проверяется.
- Докстринг `MistralDecoder` описывает «стек декодеров» с аргументом `num_layers`, хотя это один блок и такого аргумента нет; «RMSNorm перед и после» — на деле только pre-norm.
- Параметр `mask` в `GroupedQueryAttention.forward` и `MistralDecoder.forward` принимается и не используется.
- Закомментированный код: старый `PositionalEmbeddings` и `pos_out` в `Mistral`, старый блок кэширования и `_repeat_kv_heads` в `GroupedQueryAttention.forward`.
- Неиспользуемые импорты: `sqrt`, `Tensor` в `mistral.py` (`vocab_size` в `generate` ушёл вместе с копиями `generate`, пункт 18) (`k_seq_len` в `GroupedQueryAttention.forward` удалён вместе с исправлением пункта 27).
- Комментарии в `GroupedQueryAttention.forward`: сбитая нумерация шагов («Шаг 2», «3.», «5.», «8.», снова «3.», «4.») и неверные размерности (`# [B, T, hs]` там, где `[B, H, T, hs]`).
- Кэш пересобирается через `torch.cat` и срез на каждом шаге. Для учебного кода это приемлемо, но настоящего rolling buffer (запись по индексу `pos % W`) нет, хотя документация так его называет.
- ~~Нет `save`/`load` в `Mistral`~~ — есть в `BaseModel` (пункт 35).
- [`tests/models/test_mistral.py`](../llm/tests/models/test_mistral.py) проверяет только формы; генерация по одному токену с кэшем покрыта [`test_kv_cache.py`](../llm/tests/models/test_kv_cache.py). Там же — префилл кусками и генерация за `max_position_embeddings` (пункты 1, 27). Нет тестов на проверки из пункта 29 и на чтение `head_size` из конфига (пункт 28).
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: название статьи Mistral, ссылки и утверждения в докстринге `GroupedQueryAttention` (GQA — Ainslie et al., без GPT-4, без требования `num_q_heads * head_size == emb_size`) и докстринг `MistralDecoder` (один pre-norm блок) исправлены; закомментированный код и неиспользуемые импорты удалены; комментарии в `GroupedQueryAttention.forward` перенумерованы, размерности указаны 4D. Документация больше не называет кэш rolling buffer: [mistral.md](mistral.md) описывает дописывание через `torch.cat` и обрезку срезом. Сам кэш не менялся. Неиспользуемые параметры `mask` удалены из `forward` всех модулей attention и декодеров (маска паддинга проверяется в `forward` модели, пункт 3); тесты, которые передавали маску и проверяли только форму, заменены проверками встроенной causal-маски.

## Mixtral

Модель: [`models/mixtral/mixtral.py`](../llm/src/llm/models/mixtral/mixtral.py), блок: [`core/mixtral_decoder.py`](../llm/src/llm/core/mixtral_decoder.py), FFN: [`core/moe.py`](../llm/src/llm/core/moe.py) поверх [`core/swi_glu.py`](../llm/src/llm/core/swi_glu.py). Attention — тот же `GroupedQueryAttention`, что у Mistral.

Сама математика MoE верна: выход совпадает с наивным циклом по токенам (для каждого токена сумма `softmax(top-k логитов) · expert(x)`) с точностью 7e-8. Это то же, что `Softmax(TopK(x·W_g))` в статье и softmax → top-k → перенормировка в HF.

Общие с предыдущими моделями пункты касаются Mixtral так же и здесь не повторяются:
- **1**, **3** — генерация за `max_position_embeddings` и `attention_mask` (исправлены); **56** — нет поддержки левого паддинга (исправлен).
- **4**, **22** — валидация аргументов и `**kwargs` в `generate` (исправлены).
- **8** — `use_cache=True` по умолчанию и нет `no_grad` (исправлен).
- **10**, **18**, **19** — интерфейс `BaseModel`, дублирование `generate`, пограничные случаи top-k (исправлены).
- **23**, **24** — SwiGLU с `4·d` и bias во всех `Linear`. У Mixtral 8x7B эксперт — `hidden_dim = 14336` при `dim = 4096`, все проекции, включая роутер, без bias (исправлены ключами `intermediate_size` и `bias`, пункт 30).
- **25** — RoPE-буферы в `state_dict` по копии на слой (исправлен).
- **27** — нет маски при кэше и `seq_len > 1` (исправлен). До исправления префилл 6 + 8 токенов через кэш расходился с полным forward на 0.27–0.35 по логитам при `window_size=5`.
- **49** — top-p отбрасывает пограничный токен (исправлен).
- **50**, **51** — `eps` RMSNorm из конфига (исправлен, ключ `rms_norm_eps`) и dropout, которого в Mixtral 8x7B нет (задокументирован, `dropout: 0` убирает его).
- **28** — ключ `head_size` игнорировался (исправлен).
- **29** — проверки голов (исправлен).
- **33** — мусор в `GroupedQueryAttention` (исправлен).
- **31**, **32** — `_tril_mask` в `state_dict`, совместимость с torch < 1.2 (исправлены).

### Баги

#### 34. MoE падает в bf16/fp16 — P1

- **Где:** `MoE.forward`, `weights_for_expert = torch.zeros(batch_size, seq_len, device=x.device)`.
- **Что:** буфер весов создаётся без `dtype` и всегда float32. Запись в него `topk_weights[...]` в bf16/fp16 падает; обучение и инференс Mixtral в половинной точности невозможны.
- **Воспроизведено:** `MoE(16, 4, 2).to(torch.bfloat16)` на bf16-входе — `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source`.
- **Исправление:** `dtype=x.dtype`, либо переписать сборку выхода без промежуточного буфера (см. пункт 39).
- **Статус:** исправлено в ветке `fix/p1-bugs` (`dtype=x.dtype`); переписывание сборки выхода (пункт 39) не делалось. Тест: MoE в bf16/fp16 совпадает с float32 (`test_moe.py`).

#### 35. Нет `save`/`load`, хотя докстринг их обещает — P2

- **Где:** докстринг `Mixtral`: «save(path)/load(path, device) — сохранение и восстановление обученной модели».
- **Что:** методов нет ни в `Mixtral`, ни в `BaseModel`.
- **Воспроизведено:** `hasattr(Mixtral, "save")`, `hasattr(Mixtral, "load")` — `False`.
- **Исправление:** реализовать в `BaseModel` (`state_dict` + `config`, `load` как `classmethod`) — закроет и Mistral, и LLaMA. Версия для внешнего стенда уже содержит рабочий вариант с полным набором аргументов конструктора.
- **Статус:** исправлено в ветке `feat/save-load` для всех шести моделей: `model.save(path)` пишет один файл с классом модели, конфигом и `state_dict` (без вычисляемых буферов, пункты 17, 25), `Model.load(path, device)` — `classmethod`, создаёт модель по сохранённому конфигу и возвращает её в режиме `eval`. Файл читается с `weights_only=True`; файл другой модели или голый `state_dict` дают `ValueError`.

#### 36. `top_k_experts=0` принимается — P3

- **Где:** `MoE.__init__` проверяет только `top_k_experts > num_experts`.
- **Что:** при `top_k_experts=0` ни один эксперт не выбирается, FFN-ветка тождественно возвращает нули, модель молча превращается в attention-only. Отрицательное значение (`top_k_experts=-1`) конструктор тоже принимает.
- **Воспроизведено:** `Mixtral` с `top_k_experts=0` строится и выполняет `forward` без ошибок, выход MoE ровно 0.
- **Исправление:** `ValueError` при `top_k_experts < 1`.
- **Статус:** исправлено в ветке `fix/config-validation`: `MoE.__init__` требует `1 ≤ top_k_experts ≤ num_experts` и `num_experts ≥ 1`.

### Отклонения от Mixtral 8x7B

#### 37. Нет load-balancing loss у роутера — P2

- **Где:** `MoE.forward` возвращает только выход; логиты роутера наружу не отдаются, `Trainer` считает только cross-entropy.
- **Что:** статья Mixtral вспомогательный loss не описывает, но HF-реализация (`load_balancing_loss_func` в `modeling_mixtral.py`), как и Switch Transformer и GShard, добавляет при обучении `num_experts · Σ fᵢ · Pᵢ` (доля токенов на эксперта × средняя вероятность роутера). Без него роутер склонен схлопываться на пару экспертов, остальные не обучаются, и MoE вырождается в узкий dense FFN.
- **Исправление:** возвращать из `MoE` (или копить в атрибуте) `router_logits`, считать aux loss в модели с коэффициентом из конфига (`router_aux_loss_coef`, в HF `MixtralConfig` по умолчанию 0.001) и прибавлять в `Trainer`. Полезна и метрика загрузки экспертов в логах обучения.
- **Статус:** исправлено в ветке `feat/moe-load-balancing-loss`: `MoE` запоминает логиты роутера, `load_balancing_loss` (`core/moe.py`) считает формулу HF по всем слоям с учётом маски паддинга (совпадает с `load_balancing_loss_func` из `transformers` 4.57 до float, с маской и без), `Mixtral.auxiliary_loss()` умножает её на `router_aux_loss_coef`, `Trainer` и `HFGPTAdapter` прибавляют её к loss при обучении. `BaseModel.auxiliary_loss()` по умолчанию `None`. Коэффициент по умолчанию `0` — выключено, как и в HF (`output_router_logits=False`); обучение с текущими конфигами не меняется. На игрушечной задаче (8 экспертов, 300 шагов) с коэффициентом 0.02 доли загрузки экспертов — 0.10–0.15 вместо 0.06–0.18 без него, LM loss тот же. Метрика загрузки экспертов в логах `Trainer` не добавлялась.

#### 38. Двойной dropout в MoE — P2

- **Где:** `nn.Dropout` внутри каждого `SwiGLU` и ещё один на выходе `MoE`.
- **Что:** выход эксперта прорежается дважды, и эффективная вероятность выше заданной `dropout`. В Mixtral dropout в FFN нет вовсе.
- **Воспроизведено:** при `dropout=0.5` в `train()` обнуляется 62% элементов выхода MoE вместо 50% (`0.5 + 0.5 · 0.5²` для двух экспертов).
- **Исправление:** оставить один dropout — на выходе `MoE` — и создавать экспертов с `dropout=0.0` (или добавить в `SwiGLU` флаг).
- **Статус:** исправлено в ветке `fix/moe-double-dropout`: эксперты создаются с `dropout=0.0`, единственный dropout — на выходе `MoE`. При `dropout=0.5` в `train()` обнуляется 50.0% элементов выхода вместо 62.4%. Меняется только обучение: в `eval()` dropout не действует, выход побитово прежний; параметров у dropout нет, формат чекпоинтов не меняется. То, что в Mixtral dropout в FFN нет совсем, по-прежнему не учтено — это общее отличие dropout от оригиналов (пункты 51, 55).

#### 52. Sliding window attention, которого нет в Mixtral 8x7B — P2

- **Где:** `Mixtral.__init__` передаёт `window_size=config["window_size"]` в `GroupedQueryAttention`.
- **Что:** Mixtral 8x7B использует плотное внимание на весь контекст 32k («fully dense context length of 32k tokens» в статье; `sliding_window=None` в HF `MixtralConfig`). SWA — черта Mistral 7B v0.1, в Mixtral её нет. Здесь окно действует всегда.
- **Исправление:** сделать `window_size` необязательным (`None` — без окна) и по умолчанию для Mixtral не задавать; убрать ключ из `mixtral_train.json`.
- **Статус:** исправлено в ветке `feat/mistral-mixtral-hf-parity`: `window_size` необязателен в `GroupedQueryAttention`, `MistralDecoder`, `MixtralDecoder`, `Mistral` и `Mixtral`; `None` (ключа нет) — обычная causal-маска и кэш без обрезки. Ключ убран из `mixtral_train.json`. Конфиги с `window_size` работают как раньше. Сверено со случайной `MixtralForCausalLM` (`sliding_window=None`) и с `MistralForCausalLM` с окном и без; попутно тестом подтверждено, что окно здесь на позицию шире HF: `window_size = sliding_window − 1`.

#### 53. База RoPE 10 000 вместо 1 000 000 — P3

- **Где:** `RoPE(head_size, max_seq_len, base=10_000)`; ни одна модель не передаёт `base`.
- **Что:** у Mixtral 8x7B `rope_theta = 1e6` (HF `MixtralConfig`), чтобы покрыть контекст 32k. Для LLaMA-1, Mistral 7B v0.1 и Gemma 10 000 верно.
- **Исправление:** читать `config.get("rope_theta", 10_000)` и передавать в `RoPE`; для Mixtral задать 1e6 в конфигах.
- **Статус:** исправлено в ветке `feat/rope-theta`: LLaMA, Mistral, Mixtral и Gemma читают `config["rope_theta"]` (по умолчанию `10000`) и передают его базой в `RoPE` — один объект на модель, общий для всех слоёв attention. `RoPE` отклоняет базу `≤ 1`. По умолчанию выход побитово прежний; таблицы cos/sin не сохраняются в чекпоинт (пункт 25), поэтому формат не меняется. Конфиги экспериментов не менялись: для Mixtral 8x7B значение `1e6` задаётся в конфиге явно (как `rms_norm_eps` в пункте 50).

#### 54. Softmax роутера в dtype входа — P3

- **Где:** `MoE.forward`, softmax по top-k логитам роутера.
- **Что предполагалось:** HF считает `softmax(router_logits, dtype=torch.float)`, эталонный код Mistral — `softmax(weights, dtype=torch.float).to(inputs.dtype)`; здесь softmax вызывался в dtype входа, и в bf16 веса экспертов должны были терять точность.
- **Проверено:** предположение не подтвердилось. Встроенный `F.softmax` PyTorch для bf16/fp16 и так накапливает во float32, а остаток ошибки — финальное округление весов к dtype входа, которое делают и HF, и эталон. Веса в dtype входа и через float32: bf16 — совпали все 1.6 млн на CPU и 0.8 млн на MPS; fp16 — различаются 2 из 1.6 млн на CPU, совпали все на MPS; максимальное отличие от весов во float32 одинаково (bf16 — 1.95e-3, fp16 — 2.4e-4). На CUDA не проверялось; softmax там тоже накапливает во float32.
- **Исправление:** `F.softmax(topk_logits.float(), dim=-1).to(x.dtype)` — для явности и совпадения с эталоном, а не ради точности: результат не зависит от того, как softmax реализован на конкретном backend.
- **Статус:** сделано в ветке `fix/router-softmax-fp32`. На CPU и MPS выход MoE не меняется.

#### 39. Неэффективная сборка выхода MoE — P3

- **Где:** `MoE.forward`.
- **Что:** на каждого эксперта создаётся полный буфер `[batch, seq_len]` и выполняется вложенный цикл по `top_k`, токены выбираются масками сравнения. Работает, но делает лишнюю работу. (На torch < 1.2 сравнения дают uint8, и индексация uint8-маской там допустима, так что несовместимости, скорее всего, нет; запуском не проверено.)
- **Исправление:** плоский вход `[N, emb]`, `(topk_indices == e).nonzero()` даёт пары (токен, позиция в top-k), веса — `topk_weights[token_idx, k_idx]`, выход — `output.index_add_(0, token_idx, w · expert(x[token_idx]))`. Так же устроен `MixtralExperts.forward` в HF. Этот вариант уже проверен во внешнем стенде и заодно закрывает пункт 34.
- **Статус:** исправлено в ветке `perf/moe-index-add`: плоский вход `[N, emb]`, пары (токен, позиция в top-k) через `torch.where`, сборка через `index_add_`; полный буфер весов и вложенный цикл по `top_k` убраны. Выход побитово совпадает с прежним (float32 и bf16, в том числе при обучении с dropout); на 4096 токенах — 23.6 → 20.6 мс при 8 экспертах, 21.9 → 11.1 мс при 64.

#### 40. Документация и мусор — P3

- В References докстрингов нет самой статьи Mixtral — «Mixtral of Experts», Jiang et al., 2024, arXiv:2401.04088; есть только пост в блоге ([mixtral.md](mixtral.md) статью цитирует).
- Ссылка на GQA в `mixtral_decoder.py` и `mixtral.py` — `arXiv:2305.14236`, правильно `arXiv:2305.13245` (Ainslie et al.).
- Докстринг `MoE` описывает роутер как `softmax(W_r x)`, затем top-K — то есть вероятности без перенормировки. Код делает наоборот: top-K по логитам, затем softmax (что и верно, см. введение раздела).
- Неиспользуемые импорты: `Tensor`, `sqrt` в `mixtral.py`; `F` в `mixtral_decoder.py`. Параметр `mask` в `MixtralDecoder.forward` передаётся в `GroupedQueryAttention`, где игнорируется.
- Роутер создаётся с bias; в Mixtral `gate` — `Linear(dim, num_experts, bias=False)` (частный случай пункта 24).
- Тесты: [`test_moe.py`](../llm/tests/core/test_moe.py) проверяет формы, градиенты и детерминизм, но не корректность против эталона; тест на bf16/fp16 добавлен с исправлением пункта 34, префилл кусками и генерация за `max_position_embeddings` — в `test_kv_cache.py`. В [`test_mixtral.py`](../llm/tests/models/test_mixtral.py) только формы; генерация по одному токену с кэшем покрыта [`test_kv_cache.py`](../llm/tests/models/test_kv_cache.py).
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: статья Mixtral добавлена в References `Mixtral` и `MixtralDecoder`, ссылка на GQA исправлена, формула роутера в докстринге `MoE` соответствует коду, неиспользуемые импорты удалены; добавлен тест `MoE` против наивного цикла по токенам. Bias роутера остаётся частью пункта 24.

## Gemma

Модель: [`models/gemma/gemma.py`](../llm/src/llm/models/gemma/gemma.py), блок: [`core/gemma_decoder.py`](../llm/src/llm/core/gemma_decoder.py), attention: [`core/group_query_attention.py`](../llm/src/llm/core/group_query_attention.py) с [`core/rope.py`](../llm/src/llm/core/rope.py), FFN: [`core/geglu.py`](../llm/src/llm/core/geglu.py). До пункта 45 блок Gemma был построен на отдельном модуле [`core/multi_query_attention.py`](../llm/src/llm/core/multi_query_attention.py), поэтому пункты 41 и 47 касаются его; теперь он остался только учебным модулем.

Кэшированная генерация по одному токену совпадает с полным forward (покрыто [`test_kv_cache.py`](../llm/tests/models/test_kv_cache.py)).

Общие с предыдущими моделями пункты касаются Gemma так же и здесь не повторяются:
- **1**, **3** — генерация за `max_position_embeddings` и `attention_mask` (исправлены). До исправления `Gemma.forward` пропускал проверку длины при кэше, а `MultiQueryAttention` сравнивал с лимитом только `seq_len`, без `start_pos`. Неиспользуемые параметры `mask` в `GemmaDecoder.forward` и `MultiQueryAttention.forward` удалены (пункт 48): `attention_mask` проверяется в `Gemma.forward`.
- **56** — нет поддержки левого паддинга (исправлен).
- **4**, **49** — валидация аргументов `generate` и top-p (исправлены).
- **8** — `use_cache=True` по умолчанию и нет `no_grad` (исправлен).
- **10**, **18** — интерфейс `BaseModel`, дублирование `generate` (исправлены).
- **19** — пограничные случаи `generate`: `top_k` больше словаря, остановка по `eos_token_id` (исправлен).
- **20** — проверка делимости `embed_dim` на число голов (исправлен).
- **22** — `**kwargs` в `generate` (исправлен).
- **25** — RoPE-буферы в `state_dict` по копии на слой (исправлен).
- **28** — ключ `head_size` игнорировался (исправлен).
- **32** — половинчатая совместимость с torch < 1.2 в top-k/top-p `generate` и в `_tril_mask` (исправлен: проверки `hasattr(torch, "bool")` удалены). Версия Gemma на float-масках с `== 0` прошла внешний стенд 2026-09-28.
- **35** — `save`/`load`, обещанные докстрингом (исправлен).

### Баги

#### 41. Нет causal-маски при кэше и `seq_len > 1` в `MultiQueryAttention` — P1

- **Где:** `MultiQueryAttention.forward`, `if cache is None: scores = scores.masked_fill(...)`.
- **Что:** то же, что пункты 2 и 27, но в третьем модуле attention. Токены куска, поданного вместе с кэшем, видят будущее внутри куска. `generate` подаёт по одному токену, поэтому там не проявляется.
- **Воспроизведено:** префилл 4 + 6 токенов через кэш расходится с полным forward на 0.14–0.18 по логитам.
- **Исправление:** всегда накладывать маску со сдвигом `self._tril_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]` и проверять `start_pos + seq_len <= max_seq_len` (закрывает часть пункта 1). Проверено в версии для внешнего стенда: префилл кусками совпадает с полным forward.
- **Статус:** исправлено в ветке `fix/p1-bugs`.

### Отклонения от Gemma

Сравнение с Gemma 2B/7B ([Gemma Team, 2024](https://arxiv.org/abs/2403.08295); `GemmaConfig`/`GemmaModel` в HF).

#### 42. Эмбеддинги не масштабируются на √d — P2

- **Что:** в Gemma выход `embed_tokens` умножается на `sqrt(hidden_size)` перед первым блоком (в `gemma_pytorch` и старых версиях HF — `normalizer` в `GemmaModel.forward`, в текущем HF — `GemmaTextScaledWordEmbedding` с буфером `embed_scale`). Здесь эмбеддинги идут в декодер как есть.
- **Последствия:** при tied embeddings (пункт 43) без масштабирования вход в первый блок на порядок меньше по норме, чем предполагает архитектура. Веса Gemma дают неверный результат даже при совпадении остальных слоёв.
- **Исправление:** `out = tok_out * math.sqrt(embed_dim)` в `Gemma.forward` (в HF константа приводится к dtype эмбеддингов).
- **Статус:** исправлено в ветке `feat/gemma-hf-parity`: ключ `scale_embeddings` (по умолчанию `false`) умножает выход эмбеддингов на `√embed_dim`, множитель приводится к dtype эмбеддингов, как в HF. Без него сверка с HF не проходит — это проверяет тест.

#### 43. Нет weight tying, bias во всех `Linear` — P2

- **Что:** в Gemma выходная проекция привязана к `embed_tokens` (`tie_word_embeddings=True`), и все проекции без bias (`attention_bias=False`). Здесь `_linear` — отдельный `nn.Linear` с bias, bias есть в Q/K/V, выходной проекции attention и трёх матрицах GeGLU.
- **Воспроизведено:** `m._linear.bias is not None`, `m._decoders[0]._heads._q.bias is not None`, `m._linear.weight is not m._token_embeddings._embedding.weight`.
- **Последствия:** у Gemma словарь 256 000 токенов, поэтому отдельная голова — это лишние ~524M параметров для 2B (`256000 × 2048`), то есть около пятой части модели (~21% от 2.5B).
- **Исправление:** как в пунктах 5 и 24: `bias=False` под флагом конфига, `_linear.weight = _token_embeddings._embedding.weight`.
- **Статус:** исправлено в ветке `feat/gemma-hf-parity`: ключ `tie_word_embeddings` (голова без bias делит матрицу с эмбеддингами — `output_projection`, как в пункте 5) и `bias` (Q/K/V, выход attention, GeGLU и голова). По умолчанию прежняя структура, старые чекпоинты загружаются.

#### 44. GeGLU с hidden = 4·d вместо 8·d — P2

- **Где:** [`core/geglu.py`](../llm/src/llm/core/geglu.py), `nn.Linear(emb_size, 4 * emb_size)` для `_gate`, `_up`, `_down`.
- **Что:** в Gemma `intermediate_size` = 16384 при `hidden_size` = 2048 (2B) и 24576 при 3072 (7B), то есть 8·d на каждую из матриц `gate_proj` и `up_proj`. (В табл. 1 статьи «feedforward hidden dims» 32768 / 49152 — это сумма gate + up.) Здесь 4·d — FFN вдвое уже, чем в статье. Сама активация — tanh-GELU — совпадает с `gelu_pytorch_tanh` в HF. В отличие от пункта 23 (LLaMA), здесь FFN не тяжелее, а легче оригинала.
- **Исправление:** параметр `hidden_dim` в `GeGLU` с чтением из конфига, как предложено для `SwiGLU` в пункте 23.
- **Статус:** исправлено в ветке `feat/gemma-hf-parity`: `GeGLU` принимает `hidden_dim` и `bias`, `Gemma` читает ключ `intermediate_size` (по умолчанию `4 · embed_dim`; у Gemma — `8 · embed_dim`).

#### 45. Нельзя выразить Gemma 7B: MQA всегда, `head_size` = d / heads — P2

- **Что:** MQA (одна K/V-голова) используется только в Gemma 2B. Gemma 7B — обычный MHA с 16 головами, и `head_dim = 256` не равен `hidden_size / num_heads` (16 × 256 = 4096 ≠ 3072). Здесь `MultiQueryAttention` всегда с одной K/V-головой, а `head_size` всегда `embed_dim // num_q_heads` (пункт 28).
- **Исправление:** заменить `MultiQueryAttention` на `GroupedQueryAttention` с `num_kv_heads` из конфига (MQA — частный случай `num_kv_heads=1`) и читать `head_size` из конфига. `_layer` уже умеет проецировать `num_q_heads * head_size ≠ embed_dim` обратно в `embed_dim`. Тогда же уйдёт отдельный модуль MQA и пункты 41 и 47 закроются вместе с 27 и 31.
- **Статус:** исправлено в ветке `feat/gemma-hf-parity`: `GemmaDecoder` построен на `GroupedQueryAttention` без окна с `num_kv_heads` из конфига (по умолчанию `1` — MQA). При одной K/V-голове GQA транслирует её, а не копирует, поэтому результат побитово совпадает с прежним `MultiQueryAttention`, включая шаги с кэшем; кэш слоя стал `(K, V, next_pos)`. `MultiQueryAttention` остался в `llm.core` как учебный модуль. Вместе с `head_size` (пункт 28) Gemma 7B — `num_kv_heads: 16`, `head_size: 256` при `embed_dim: 3072` — выразима.

#### 46. RMSNorm без `(1 + w)` и вычислений во float32 — P3

- **Где:** [`core/rms_norm.py`](../llm/src/llm/core/rms_norm.py).
- **Что:** в Gemma `GemmaRMSNorm` хранит вес, инициализированный нулями, и умножает на `(1 + weight)`, а нормализацию считает во float32 и приводит результат обратно. Здесь вес инициализирован единицами и умножается напрямую, вычисления в dtype входа. При обучении с нуля параметризации эквивалентны, но веса Gemma без поправки `+1` не загрузятся корректно, а в bf16 нормализация менее точна.
- **Исправление:** для загрузки весов — прибавлять 1 при конвертации. Для bf16 — `x.float()` внутри `forward` и `.to(x.dtype)` на выходе (затрагивает LLaMA, Mistral, Mixtral).
- **Замечание по загрузке весов HF в целом:** помимо пунктов 42–46 нужна перестановка строк `q_proj`/`k_proj` — HF Gemma использует `rotate_half`, здесь чередующиеся пары (как в пункте 24).
- **Статус:** исправлено в ветке `feat/gemma-hf-parity`: `RMSNorm` для float16/bfloat16 считает нормализацию во float32 и приводит к dtype входа перед умножением на вес, как `LlamaRMSNorm` (затрагивает LLaMA, Mistral, Mixtral и Gemma; во float32 побитово прежний результат). Параметризация `(1 + w)` не добавлялась: `convert_hf_state_dict` из `llm.models.gemma` прибавляет 1 к весам RMSNorm. Веса `GemmaForCausalLM` загружаются; сверено со случайными моделями HF в форме 2B (MQA) и 7B (MHA, `head_dim` ≠ `hidden / heads`): логиты до ~1e-5, greedy с KV-кэшем совпадает. Остаётся отличие в bf16 в последних битах: `GemmaRMSNorm` умножает на вес ещё во float32.

#### 55. Dropout на эмбеддингах, в attention и GeGLU — P3

- **Что:** в Gemma dropout нет (`attention_dropout=0.0` в HF, в `gemma_pytorch` его нет). Здесь dropout стоит после эмбеддингов (`Gemma.forward`), в `MultiQueryAttention` и в `GeGLU`.
- **Исправление:** ставить `dropout=0.0` по умолчанию; то же для Mistral (пункт 51).
- **Статус:** сделано в ветке `docs/mistral-gemma-dropout`: в [gemma.md](gemma.md#отличия-от-gemma) и таблице конфигурации указано, что dropout в оригинале нет и `dropout: 0` убирает его полностью (проверено: все пять dropout модели получают `p = 0`). Значения по умолчанию нет: `dropout` — обязательный ключ конфига.

### Качество кода

#### 47. `_tril_mask` в `state_dict` — P2

- **Где:** `MultiQueryAttention.__init__`, `register_buffer("_tril_mask", ...)`.
- **Что:** то же, что пункты 17 и 31, но в `MultiQueryAttention`, поэтому их исправление его не закроет.
- **Воспроизведено:** ключи `_decoders.{i}._heads._tril_mask` в `state_dict`.
- **Исправление:** `persistent=False`.
- **Статус:** исправлено в ветке `fix/nonpersistent-buffers` (как пункт 17).

#### 48. Документация и мусор — P3

- Докстринги `Gemma` и `GemmaDecoder` описывают несуществующие варианты: «Multi-Query либо Grouped heads», «FFN с GeGLU/SwiGLU», «RMSNorm или LayerNorm»; псевдокод в `GemmaDecoder` использует `LayerNorm`. В коде всегда MQA + GeGLU + RMSNorm.
- Неверная ссылка на статью Gemma в докстрингах `Gemma`, `Gemma.generate` и `GemmaDecoder`: `arXiv:2403.07794`, правильно `arXiv:2403.08295`.
- Неиспользуемые импорты: `math`, `sqrt`, `Tensor` в `gemma.py`; `F` в `gemma_decoder.py`.
- Комментарии в `MultiQueryAttention.forward`: сбитая нумерация шагов («Шаг 2», «3.», «5.», снова «3.», «4.») и неверные размерности (`# [B, T, hs]` там, где `[B, H, T, hs]`).
- [`gemma_train.json`](../experiments/llm_only/configs/gemma_train.json) содержит ключи Mixtral (`num_kv_heads`, `num_experts`, `top_k_experts`, `window_size`), которые модель не читает. Ключи убраны из JSON.
- Тесты: [`test_gemma.py`](../llm/tests/models/test_gemma.py) проверяет только формы. ~~`test_forward_masked` в `test_gemma_decoder.py` создавал впечатление, что блок применяет маску~~ — заменён проверкой встроенной causal-маски. Префилл кусками и генерация за `max_position_embeddings` покрыты `test_kv_cache.py`.
- **Статус:** исправлено в ветке `chore/docs-and-cleanup`: докстринги `Gemma` и `GemmaDecoder` описывают фактическую схему (MQA + GeGLU + RMSNorm), ссылка на статью Gemma исправлена, неиспользуемые импорты удалены, комментарии в `MultiQueryAttention.forward` перенумерованы. Ключи Mixtral убраны из `gemma_train.json`, раздел о неиспользуемых ключах в [gemma.md](gemma.md) удалён. Неиспользуемые параметры `mask` удалены из `forward` всех модулей attention и декодеров (маска паддинга проверяется в `forward` модели, пункт 3); тесты, которые передавали маску и проверяли только форму, заменены проверками встроенной causal-маски.


## Токенизатор, данные и обучение

Пункты 57–62 найдены при написании глав [tokenization.md](tokenization.md) и [training.md](training.md) учебного пособия (2026-09-29) и проверены запуском.

#### 57. Паддинг входит в loss — P1

- **Где:** `TextDataset`, `TextWithSpecialTokensDataset`, `StreamingTextDataset` (`llm/src/llm/datasets/`) и `Trainer.compute_lm_loss`.
- **Что:** датасеты дополняют последовательность до `block_size` токеном `pad_token_id` и возвращают `labels = input_ids.clone()`, то есть pad-позиции остаются в метках. `F.cross_entropy(..., ignore_index=-100)` в `Trainer` их не отбрасывает, хотя комментарий обещает, что «padding токены не участвуют в loss». Метку `-100` ставит только коллатор hf-proxy.
- **Воспроизведено:** на учебном корпусе `experiments/llm_only` при `block_size = 128` около 94 % целей — предсказание pad после pad. Валидационный loss без маски — 1.23, с метками `-100` на паддинге — 5.92 при `ln V = 6.18` (токенизатор, обученный на всех строках корпуса, V = 484): модель в основном учится повторять pad, а заниженный loss это скрывает.
- **Исправление:** в `labels` заменять паддинг на `-100` (`labels[attention_mask == 0] = -100` или прямо при дополнении). Заодно возвращать `attention_mask` из датасетов.
- **Статус:** исправлено в ветке `fix/pad-in-loss`: все три датасета собирают пример функцией `lm_example` ([`datasets/lm_example.py`](../llm/src/llm/datasets/lm_example.py)) и возвращают `input_ids`, `attention_mask` и `labels` с `-100` на паддинге. Паддинг определяется по месту, а не по значению токена: `pad_token_id` может совпадать с настоящим токеном (0 по умолчанию, pad = EOS), и такие токены остаются в loss. `Trainer` передаёт `attention_mask` из батча в модель (у Mixtral паддинг больше не входит в статистику роутера), а на батче без единой цели `compute_lm_loss` возвращает 0 вместо NaN. Тесты: маска и метки у всех датасетов, в том числе при pad = настоящему токену и pad = EOS; loss `Trainer` на GPT не зависит от `block_size`; маска доходит до Mixtral.

#### 58. `BPETokenizer.encode` не применяет слияния по порядку — P2

- **Где:** `BPETokenizer.encode` (`llm/src/llm/tokenizers/bpe_tokenizer.py`).
- **Что:** слово кодируется жадным поиском самого длинного токена из словаря, начиная с текущего символа (как WordPiece), а список `merges` при кодировании не используется. BPE (Sennrich et al., 2016) кодирует новое слово, применяя выученные слияния в порядке их появления. Результаты расходятся, а поиск перебирает весь `vocab_list` на каждой позиции — O(n·V).
- **Воспроизведено:** на корпусе из статьи Sennrich (low×5, lower×2, newest×6, widest×3) слово `nest` по слияниям кодируется как `n est`, здесь — `ne s t`.
- **Исправление:** кодировать слово применением `merges` по рангу (как в [tokenization.md](tokenization.md)); жадный поиск оставить разве что как отдельный режим.
- **Статус:** исправлено в ветке `fix/bpe-merges-encode`: `encode` разбивает слово методом `_bpe_word` — слияния по порядку ранга, как `bpe()` в GPT-2; разбиение слова кэшируется на время вызова. Жадный longest-match (`_greedy_word`) остался только для токенизатора без слияний (старые файлы `save` и `save_pretrained` hf-proxy без поля `merges`): по словарю порядок слияний не восстановить. `HFTokenizerAdapter.save_pretrained` теперь сохраняет `merges`, `from_pretrained` их загружает — раньше загруженный адаптер кодировал бы по одному словарю. Тесты: `nest` → `n est`, слова корпуса разбиваются как при обучении, совпадение с BPE из HuggingFace `tokenizers` на всех словах длины 1–4 из алфавита примера Sennrich и 2 000 случайных словах длины 5–9, сохранение слияний в hf-proxy. На учебном корпусе `experiments` (словарь 1000) оба способа кодируют все слова обучающей и валидационной выборок и тестовые промпты одинаково: каждое слово слилось в один токен.

#### 59. Неизвестный символ без `<unk>` даёт `None` — P2

- **Где:** `BPETokenizer.encode`, `decode`.
- **Что:** если `<unk>` не передан в `special_tokens` при обучении, `unk_token_id` равен `None`, и `encode` возвращает `None` на месте неизвестного символа — модель упадёт на таком входе. Если `<unk>` есть, `decode` по умолчанию молча выбрасывает его. В учебных конфигах `test_prompts` частично английские при русском корпусе `TRAIN_TEXTS` и кодируются почти целиком в `<unk>`.
- **Исправление:** всегда добавлять `<unk>` в словарь (или бросать `ValueError` на неизвестный символ); промпты конфигов привести к языку корпуса.
- **Статус:** исправлено в ветке `fix/bpe-unknown-symbol` вторым способом: без `<unk>` в словаре `encode` бросает `ValueError` со списком неизвестных символов. Словарь и id не меняются, поэтому обученные модели и сохранённые токенизаторы не затронуты. `test_prompts` всех конфигов `experiments/llm_only` заменены русскими фразами из символов обучающей части корпуса; в них были и английские промпты, и символы, которых нет в корпусе, в русских (`—`, `-`, `2`). В `TEST_PROMPTS` из `experiments/shared/configs.py` (промпты HF-экспериментов) `"Программирование"` с кириллической `П`, которой нет в корпусе, заменено на `"Мир"`. `decode` по-прежнему выбрасывает `<unk>` при `skip_special_tokens=True` — как HuggingFace; это описано в [tokenization.md](tokenization.md).

#### 60. Warmup длиннее всего обучения в учебных конфигах — P3

- **Где:** `experiments/llm_only/configs/*_train.json`: `warmup_steps = 50` при `num_epochs = 3`.
- **Что:** на учебном корпусе за три эпохи получается около 18 шагов, и learning rate не поднимается выше ≈ 0.34 от заданного — всё обучение проходит внутри warmup. Первый шаг `LambdaLR` делается с `lr = 0`.
- **Воспроизведено:** средний loss по эпохам с `warmup_steps = 5` — 5.34 → 1.33, с 50 — 6.07 → 3.74 (loss с учётом паддинга, пункт 57).
- **Исправление:** задавать warmup долей от числа шагов (например, 5–10 %) или уменьшить значение в конфигах.
- **Статус:** исправлено в ветке `fix/warmup-steps`: `Trainer` принимает `warmup_ratio` — долю warmup от числа шагов, $`\lceil N_{\text{steps}} \cdot \texttt{warmup\_ratio} \rceil`$, как в HF `TrainingArguments` (вместе с `warmup_steps` — `ValueError`; без обоих — 100 шагов, как раньше), и предупреждает, если warmup не короче всего обучения. Учебные конфиги `experiments/llm_only` и `TRAINING_CONFIG` в `experiments/shared/configs.py` (там была та же ошибка, его использует `train_with_hf_trainer.py`) переведены на `warmup_ratio = 0.1`: 2 шага warmup из 18, learning rate доходит до заданного. Прогон GPT с паддингом, исключённым из loss (пункт 57): средний loss эпох 6.08 → 5.98 → 5.79 с `warmup_steps = 50` и 6.04 → 5.27 → 4.84 с `warmup_ratio = 0.1`. Первый шаг с `lr = 0` оставлен: так же устроен линейный warmup в HF.

#### 61. `get_optimizer`: weight decay на всех параметрах — P3

- **Где:** `get_optimizer` (`llm/src/llm/training/optimizer.py`).
- **Что:** `AdamW(model.parameters(), weight_decay=0.01)` затухает все параметры, включая bias, веса нормализаций и эмбеддинги; обычно (GPT-2, LLaMA, HF `Trainer`) их исключают. Вариант `"adam"` — L2-регуляризация, а не decoupled decay; в варианте `"sgd"` `weight_decay` игнорируется.
- **Исправление:** две группы параметров — с decay для матриц `Linear` и без decay для bias, норм и эмбеддингов; описать поведение `"adam"` и `"sgd"` в докстринге.

#### 62. LLaMA, Mistral, Mixtral и Gemma без инициализации из статей — P3

- **Где:** конструкторы `Llama`, `Mistral`, `Mixtral`, `Gemma`; `init_normal_` (`core/weight_init.py`) вызывают только `GPT` и `GPT2` (пункты 7, 16).
- **Что:** остальные модели используют инициализацию PyTorch по умолчанию: эмбеддинги N(0, 1), `Linear` — равномерное с std ≈ 1/√(3·fan_in). В HF-конфигах этих моделей `initializer_range = 0.02`. На загрузку весов HF это не влияет, только на обучение с нуля.
- **Воспроизведено:** начальный loss свежих моделей при V = 1000 — 7.05–7.09 против 6.96 у GPT. Gemma с `tie_word_embeddings` и `scale_embeddings` даёт начальный loss ≈ 258 вместо ln V ≈ 6.9: эмбеддинги N(0, 1), умноженные на √d, и та же матрица на выходе.
- **Исправление:** применять `init_normal_` с `initializer_range` (по умолчанию 0.02) во всех моделях; для Gemma — обязательно при `scale_embeddings`.
