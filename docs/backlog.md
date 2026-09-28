# Бэклог

Технический долг, найденный при разборе кода. Каждая запись: где проблема, как её воспроизвести, как исправить.

Приоритеты: **P1** — неверный результат или падение; **P2** — расхождение с документацией или статьёй, дешёвые исправления; **P3** — качество кода.

Состояние кода — ветка `fix/kv-cache` на 2026-09-28. Пункты с пометкой «воспроизведено» проверены запуском.

## GPT-1

Модель: [`models/gpt/gpt.py`](../llm/src/llm/models/gpt/gpt.py), блок: [`core/gpt_decoder.py`](../llm/src/llm/core/gpt_decoder.py). Пункты 2, 3, 8 и 10 касаются общих модулей и затрагивают и другие архитектуры.

### Баги

#### 1. `generate` падает за пределами `max_position_embeddings` — P1

- **Где:** `GPT.generate`, `GPT.forward`, `PositionalEmbeddings.forward`.
- **Что:** контекст не обрезается. Проверки длины смотрят на `seq_len`, а не на `start_pos + seq_len`.
- **Воспроизведено:** `max_position_embeddings=16`, промпт 10 токенов, `max_new_tokens=10`. С кэшем — `IndexError: index out of range in self` из `nn.Embedding` (на CUDA — device-side assert). Без кэша — `ValueError`.
- **Исправление:** при `x.size(1) > max_seq_len` обрезать окно `x[:, -max_seq_len:]` и пересчитывать без кэша (абсолютные позиции сдвигаются, кэш становится невалидным). Проверять `start_pos + seq_len` в `forward` и `PositionalEmbeddings`.
- **Статус:** общая часть упомянута в [известных ограничениях](README.md#известные-ограничения).

#### 2. Нет causal-маски при кэше и `seq_len > 1` — P1

- **Где:** [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py), `if cache is None: scores = scores.masked_fill(...)`.
- **Что:** если передать кэш и несколько токенов (префилл кусками, спекулятивное декодирование), будущие токены внутри куска не маскируются. `generate` подаёт по одному токену, поэтому там не проявляется.
- **Воспроизведено:** префилл 4 + 6 токенов через кэш расходится с полным forward на 0.21 по логитам.
- **Исправление:** всегда накладывать маску со сдвигом `self._tril_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]`.
- **Статус:** упомянуто в [известных ограничениях](README.md#известные-ограничения).

#### 3. `attention_mask` молча игнорируется — P1

- **Где:** принимается в `GPT.forward`, `GptDecoder.forward`, `MultiHeadAttention.forward`, но нигде не применяется. `hf-proxy/src/hf_proxy/hf_adapter.py` передаёт маску, ожидая, что она сработает.
- **Воспроизведено:** `attention_mask` из нулей даёт логиты, побитово равные вызову без маски.
- **Исправление:** пробросить маску до attention и накладывать её вместе с causal-маской (`[B, T]` → `[B, 1, 1, T_kv]`). Либо, пока не реализовано, убрать параметр или бросать `NotImplementedError`, чтобы не было молчаливой ошибки.
- **Статус:** упомянуто в [известных ограничениях](README.md#известные-ограничения).

#### 4. `generate` не валидирует аргументы, хотя докстринг обещает — P2

- **Где:** `GPT.generate`.
- **Что:** в докстринге описаны `ValueError` при `temperature ≤ 0`, одновременных `top_k` и `top_p`, `top_k ≤ 0`, `top_p ∉ (0, 1]`. В коде проверок нет.
- **Воспроизведено:** `temperature=0.0` и `top_k=5, top_p=0.9` принимаются молча. При `temperature=0` масштабирование просто пропускается.
- **Исправление:** добавить проверки из докстринга в начало метода.

### Отклонения от GPT-1

#### 5. Нет weight tying — P2

- **Что:** в оригинальном коде OpenAI и в HuggingFace (`OpenAIGPTLMHeadModel.lm_head`, без bias, привязан к `tokens_embed`) выходная проекция делит веса с токенными эмбеддингами. Здесь `_linear` — отдельный `nn.Linear` с bias.
- **Последствия:** примерно на `vocab_size × embed_dim` параметров больше; веса `openai-community/openai-gpt` напрямую не загружаются.
- **Исправление:** `_linear = nn.Linear(embed_dim, vocab_size, bias=False)` и `_linear.weight = _token_embeddings._embedding.weight`, под флагом конфига, если нужна обратная совместимость чекпойнтов.

#### 6. Нет dropout на весах внимания — P3

- **Что:** в GPT-1 (`attn_pdrop=0.1` в HF) dropout применяется к весам после softmax. Здесь есть только dropout после выходной проекции.
- **Исправление:** `weights = self._attn_dropout(F.softmax(scores, dim=-1))`, отдельным параметром.

#### 7. Нет инициализации весов из статьи — P3

- **Что:** в статье веса инициализируются N(0, 0.02). В репозитории используется инициализация PyTorch по умолчанию.
- **Исправление:** метод `_init_weights` (Linear/Embedding — `normal_(0, 0.02)`, bias — нули) и вызов `self.apply(...)` в `__init__`.

### Качество кода

#### 8. `use_cache=True` по умолчанию и нет `torch.no_grad()` в `generate` — P2

- **Что:** при обучении `forward` возвращает ненужные K/V каждого слоя. `generate` без `no_grad` у вызывающего строит autograd-граф на всю генерацию.
- **Воспроизведено:** в `eval()` логиты имеют `requires_grad=True`, кэш возвращается по умолчанию.
- **Исправление:** `use_cache=False` по умолчанию в `forward` (проверить `Trainer` и `hf_adapter`), декоратор `@torch.no_grad()` на `generate`.

#### 9. Приведение dtype внутри `FeedForward.forward` — P3

- **Где:** [`core/feed_forward.py`](../llm/src/llm/core/feed_forward.py).
- **Что:** `_layer1`/`_layer2` переприсваиваются во время forward, если dtype входа отличается. Это скрывает ошибки dtype и рассинхронизирует состояние оптимизатора.
- **Исправление:** убрать, приводить модель снаружи (`model.to(dtype)`) или использовать `torch.autocast`.

#### 10. Интерфейс `BaseModel` не соответствует моделям — P3

- **Где:** [`core/base_model.py`](../llm/src/llm/core/base_model.py).
- **Что:** объявлены `forward(input_ids, attention_mask) -> Tensor` и `generate(input_ids, max_length)`; `GPT` возвращает `(logits, cache)` и принимает `max_new_tokens`, `do_sample` и т.д.
- **Исправление:** привести абстрактные сигнатуры к фактическим.
- **Статус:** упомянуто в [известных ограничениях](README.md#известные-ограничения).

#### 11. Документация противоречит коду — P2

- Докстринг `GptDecoder` называет блок «pre-LN» и приводит pre-LN псевдокод; в коде post-LN.
- Пример в докстринге `GptDecoder` использует `Decoder(...)` и ожидает от `decoder(x)` тензор, а возвращается кортеж.
- В References класса `GPT` битая ссылка на статью: `research-covers/languageunsupervised/` (нет дефиса, правильно `language-unsupervised`).

#### 12. Мусор в коде — P3

- Закомментированный старый `generate` в конце `gpt.py`.
- Неиспользуемые импорты: `Optional`, `Dict` в `gpt.py`, `math` в `feed_forward.py`.
- Мёртвые проверки `hasattr(torch, "bool")` (актуальны только для PyTorch < 1.2).
- Сравнения `do_sample == True`, `top_k != None` вместо `if do_sample`, `is not None`.

## GPT-2

Модель: [`models/gpt/gpt2.py`](../llm/src/llm/models/gpt/gpt2.py), блок: [`core/gpt2_decoder.py`](../llm/src/llm/core/gpt2_decoder.py).

Общие с GPT-1 пункты касаются GPT-2 так же и здесь не повторяются:
- **1** — падение `generate` за `max_position_embeddings`. Воспроизведено на GPT-2 с `max_position_embeddings=16`, промптом 10 и `max_new_tokens=10`: с кэшем `IndexError`, без кэша `ValueError`.
- **2** — нет causal-маски при кэше и `seq_len > 1`. На GPT-2 префилл 4 + 6 расходится с полным forward на 0.12.
- **3** — игнорируется `attention_mask`.
- **4** — нет валидации аргументов `generate`.
- **8** — `use_cache=True` по умолчанию и нет `no_grad`.
- **9** — dtype в `FeedForward`.
- **10** — интерфейс `BaseModel`.
- **12** — мёртвые проверки `hasattr(torch, "bool")` и сравнения `== True` / `!= None`.

### Отклонения от GPT-2

#### 13. GELU: точная erf-версия вместо tanh-аппроксимации — P2

- **Что:** `Gpt2Decoder` создаёт `FeedForward(activation="gelu")`, а это `nn.GELU()`, то есть erf. Оригинальный код OpenAI (`gpt-2/src/model.py`) и HF (`GPT2Config.activation_function="gelu_new"`) используют tanh-аппроксимацию. Кроме того, опция `'gelu_exact'` в `FeedForward` на деле подключает tanh-аппроксимацию, то есть название обратно смыслу. То же касается GPT-1: там тоже tanh (`finetune-transformer-lm/train.py`).
- **Воспроизведено:** при одинаковых весах логиты отличаются от эталона с tanh-GELU на ~1e-4. С tanh-GELU расхождение 5e-7.
- **Исправление:** переименовать `'gelu_exact'` → `'gelu_tanh'`, в `GptDecoder` и `Gpt2Decoder` передавать `activation="gelu_tanh"`.
- **Статус:** исправлено в ветке `fix/gelu-tanh` (коммит `32a2db5`), в `master` не влито.

#### 14. Нет weight tying, у lm-head есть bias — P2

- **Что:** в оригинале и в HF (`GPT2LMHeadModel.lm_head`, `bias=False`, `tie_word_embeddings=True`) выходная проекция делит веса с `wte`. Здесь `_linear` — отдельный `nn.Linear` с bias.
- **Воспроизведено:** `m._linear.bias is not None`, `m._linear.weight is not m._token_embeddings._embedding.weight`.
- **Последствия:** для конфигурации 124M лишних ~38M параметров (`50257 × 768`). Веса `openai-community/gpt2` напрямую не загружаются.
- **Исправление:** как в пункте 5 для GPT-1.

#### 15. Нет dropout на весах внимания — P3

- **Что:** в GPT-2 (`attn_pdrop=0.1` в HF) dropout применяется к весам после softmax. Здесь только dropout после выходной проекции (`resid_pdrop`).
- **Исправление:** общее с пунктом 6, так как `MultiHeadAttention` общий.

#### 16. Нет инициализации весов из статьи — P3

- **Что:** GPT-2 инициализирует веса N(0, 0.02), а выходные проекции residual-веток (`c_proj` в attention и MLP) масштабирует на `1/√(2·num_layers)` (разд. 2.3 статьи; `GPT2PreTrainedModel._init_weights` в HF). В репозитории инициализация PyTorch по умолчанию.
- **Исправление:** как в пункте 7, плюс `normal_(0, 0.02 / math.sqrt(2 * num_layers))` для `MultiHeadAttention._layer` и `FeedForward._layer2`.

### Качество кода

#### 17. `_tril_mask` сохраняется в `state_dict` — P2

- **Где:** [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py), `register_buffer('_tril_mask', ...)`. Затрагивает все модели на `MultiHeadAttention`.
- **Что:** буфер `max_seq_len × max_seq_len` persistent: попадает в каждый чекпоинт по одному на слой и привязывает чекпоинт к `max_seq_len`.
- **Воспроизведено:** ключи `_decoders.{i}._heads._tril_mask` в `state_dict`. При `max_position_embeddings=1024` это 1 МБ на слой.
- **Исправление:** `register_buffer(..., persistent=False)`. Старые чекпоинты при этом загружаются только с `strict=False` или после удаления ключей.

#### 18. `generate` скопирован в шесть моделей — P2

- **Где:** `generate` в `gpt.py`, `gpt2.py`, `llama.py`, `mistral.py`, `mixtral.py`, `gemma.py`.
- **Что:** логика temperature/top-k/top-p/sampling одинакова, поэтому каждое исправление (пункты 1, 4, 19, `hasattr(torch, "bool")`) нужно вносить шесть раз.
- **Исправление:** вынести выбор следующего токена в общую функцию (например, `core/sampling.py`) или в `BaseModel.generate` поверх `forward(x, use_cache, cache)`.

#### 19. Пограничные случаи в `generate` — P3

- **Что:**
  - `top_k > vocab_size` падает в `torch.topk`;
  - нет остановки по `eos_token_id`, всегда генерируется ровно `max_new_tokens`.
- **Воспроизведено:** `top_k=100` при `vocab_size=50` — `RuntimeError: selected index k out of range`.
- **Исправление:** `top_k = min(top_k, vocab_size)`; параметр `eos_token_id` с остановкой, когда все последовательности батча его сгенерировали.

#### 20. `head_size` без проверки делимости — P3

- **Где:** `GPT2.__init__`, `head_size=config["embed_dim"] // config["num_heads"]`.
- **Что:** при `embed_dim % num_heads != 0` размер головы молча усекается, и внимание работает в пространстве меньше `embed_dim`.
- **Воспроизведено:** `embed_dim=30, num_heads=4` принимается, `head_size=7`, Q/K/V — 28 измерений.
- **Исправление:** `assert`/`ValueError` в `__init__`. Связано с тем, что ключ `head_size` в конфигах не читается (см. [известные ограничения](README.md#известные-ограничения)).

#### 21. Документация и мусор — P3

- Пример в докстринге модуля: `GPT2({"vocab_size": 50257, ...})` и `model.generate(input_ids, max_length=30)`. Воспроизведено: `TypeError`, нет обязательных `max_new_tokens` и `do_sample`.
- Неиспользуемые импорты в `gpt2.py`: `FeedForward`, `Tensor`.
- Параметр `rope` в `Gpt2Decoder` (и импорт `RoPE`) — GPT-2 его не использует.

## LLaMA

Модель: [`models/llama/llama.py`](../llm/src/llm/models/llama/llama.py), блок: [`core/cached_decoder.py`](../llm/src/llm/core/cached_decoder.py), attention: [`core/multi_head_attention.py`](../llm/src/llm/core/multi_head_attention.py) с [`core/rope.py`](../llm/src/llm/core/rope.py).

Общие с GPT пункты касаются LLaMA так же и здесь не повторяются:
- **1** — падение `generate` за `max_position_embeddings`. Воспроизведено с `max_position_embeddings=16`, промптом 10 и `max_new_tokens=10`: с кэшем `RuntimeError: shape '[1, 1, 1, 8]' is invalid for input of size 0` из `RoPE.forward` (пустой срез cos/sin), без кэша `ValueError`. В моделях с RoPE контекст нельзя просто обрезать окном и продолжить с кэшем: при обрезке нужно пересчитать K заново с новыми позициями.
- **2** — нет causal-маски при кэше и `seq_len > 1`. Префилл 4 + 6 расходится с полным forward на 0.28.
- **3** — игнорируется `attention_mask`. У `Llama.forward` такого параметра нет вовсе, а `generate` его принимает: `attention_mask` из нулей даёт тот же результат, что и без маски.
- **4** — нет валидации аргументов `generate`. `temperature=0.0` и `top_k=5, top_p=0.9` принимаются молча.
- **8** — `use_cache=True` по умолчанию и нет `no_grad`. В `eval()` логиты имеют `requires_grad=True`.
- **10** — интерфейс `BaseModel`.
- **12** — мёртвые проверки `hasattr(torch, "bool")` (в `generate` LLaMA — тройные тернарники прямо в строках) и сравнения `== True` / `!= None`.
- **17** — `_tril_mask` в `state_dict`: `MultiHeadAttention` общий.
- **18** — дублирование `generate`.
- **19** — `top_k=100` при `vocab_size=50` падает с `RuntimeError: selected index k out of range`.
- **20** — проверка делимости. Воспроизведено: `embed_dim=100, num_heads=6` принимается, Q/K/V — 96 измерений. При нечётном `head_size` (`embed_dim=30, num_heads=4`) падает `assert` в `RoPE.__init__` с сообщением «head_size должен быть четным» — без упоминания `embed_dim` и `num_heads`.

### Баги

#### 22. `generate` молча принимает любые именованные аргументы — P2

- **Где:** `Llama.generate(..., attention_mask=None, **kwargs)`.
- **Что:** `**kwargs` нигде не используется, поэтому опечатки и аргументы из других API (`max_length`, `eos_token_id`) проглатываются без ошибки.
- **Воспроизведено:** `generate(x, 2, do_sample=False, max_lenght=5)` выполняется без ошибок.
- **Исправление:** убрать `**kwargs`. Если он нужен для совместимости с `hf_adapter`, явно перечислить поддерживаемые ключи и бросать `TypeError` на остальных.

### Отклонения от LLaMA

Докстринг `Llama` и [llama.md](llama.md#известное-расхождение-с-докстрингом) уже упоминают bias и dropout; ниже — что из этого следует и чего там нет.

#### 23. SwiGLU с hidden = 4·d вместо ⅔·4·d — P2

- **Где:** [`core/swi_glu.py`](../llm/src/llm/core/swi_glu.py), `nn.Linear(emb_size, 4 * emb_size)` для `_gate`, `_up`, `_down`.
- **Что:** в LLaMA (разд. 2.3 статьи; `FeedForward` в `facebookresearch/llama/model.py`) скрытая размерность — `2/3 · 4d`, округлённая вверх до кратного `multiple_of=256`, чтобы три матрицы SwiGLU весили столько же, сколько две матрицы обычного FFN с `4d`. Здесь три матрицы по `4d`.
- **Последствия:** FFN примерно в 1.5 раза тяжелее, чем в статье. Для `d=4096`: hidden 16384 вместо 11008, ~201M вместо ~135M параметров FFN на слой. При сравнении с GPT той же ширины LLaMA получает лишние параметры, и сравнение архитектур становится нечестным.
- **Исправление:** параметр `hidden_dim` в `SwiGLU` (по умолчанию — формула LLaMA, опционально `multiple_of`). Затрагивает Mistral и Mixtral, которые используют тот же `SwiGLU`; меняет размеры весов, поэтому старые чекпоинты не загрузятся.
- **Статус:** не задокументировано.

#### 24. Bias во всех `Linear` — P3

- **Что:** в LLaMA все проекции (`wq`, `wk`, `wv`, `wo`, `w1`–`w3`, `output`) без bias. Здесь bias есть в Q/K/V, выходной проекции attention, трёх матрицах SwiGLU и голове на словарь.
- **Воспроизведено:** `m._decoders[0]._heads._q.bias is not None`, `m._linear.bias is not None`.
- **Последствия:** веса Meta/HF LLaMA напрямую не загружаются (лишние ключи `*.bias`). Для загрузки весов HF, помимо bias, нужна перестановка строк `q_proj`/`k_proj`: HF использует `rotate_half` (половины вектора), а здесь, как у Meta, — чередующиеся пары `(2i, 2i+1)`.
- **Исправление:** флаг `bias` в конфиге (по умолчанию `False` для LLaMA) с пробросом в `MultiHeadAttention` и `SwiGLU`.
- **Статус:** задокументировано в докстринге и [llama.md](llama.md#известное-расхождение-с-докстрингом).

### Качество кода

#### 25. RoPE-буферы в `state_dict`, по копии на каждый слой — P2

- **Где:** [`core/rope.py`](../llm/src/llm/core/rope.py), `register_buffer("cos_matrix", ...)` и `register_buffer("sin_matrix", ...)`. Один объект `RoPE` зарегистрирован в модели и в `MultiHeadAttention` каждого слоя.
- **Что:** буферы persistent, и `state_dict` содержит их под `num_layers + 1` ключами: `_position_embeddings.cos_matrix`, `_decoders.{i}._heads._rope.cos_matrix` и т.д. Чекпоинт хранит одни и те же таблицы многократно и привязан к `max_position_embeddings` — увеличить контекст без правки `state_dict` нельзя.
- **Воспроизведено:** при `num_layers=2` — три ключа `*.cos_matrix` и три `*.sin_matrix`.
- **Исправление:** `persistent=False` (как в пункте 17). Затрагивает все модели с RoPE: Mistral, Mixtral, Gemma.

#### 26. Документация и мусор — P3

- Закомментированный блок вычисления `start_pos` и строка `# pos_out = ...` в `Llama.forward`; неиспользуемая переменная `vocab_size` в `generate`.
- Неиспользуемые импорты: `FeedForward` в `cached_decoder.py`, `Optional` в `rope.py`, `swi_glu.py`, `rms_norm.py`.
- Докстринг `CachedDecoder` описывает LayerNorm и GELU, хотя для LLaMA блок собирается с `RMSNorm` и `SwiGLU`.
- Комментарий к форме выхода в `RoPE.forward` — `[batch_size, seq_len, head_size]`, фактически 4D `[batch, num_heads, seq_len, head_size]`.
- В [README.md](README.md) устарели пометки «⚠️ без GQA, вопреки докстрингу» в таблице и пункт «LLaMA — нет GQA, вопреки докстрингу» в известных ограничениях: докстринг уже исправлен, расхождения больше нет.
- Нет `save`/`load` ни в `Llama`, ни в `BaseModel`, хотя версия для внешнего стенда их требует.
- [`tests/models/test_llama.py`](../llm/tests/models/test_llama.py) проверяет только формы. Нет тестов на префилл кусками с кэшем (пункт 2), на генерацию до границы `max_position_embeddings` (пункт 1) и на то, что top-k/top-p оставляют нужное число токенов.

## Mistral

Модель: [`models/mistral/mistral.py`](../llm/src/llm/models/mistral/mistral.py), блок: [`core/mistral_decoder.py`](../llm/src/llm/core/mistral_decoder.py), attention: [`core/group_query_attention.py`](../llm/src/llm/core/group_query_attention.py) с [`core/rope.py`](../llm/src/llm/core/rope.py). `GroupedQueryAttention` общий с Mixtral, поэтому пункты 27–30 затрагивают и её.

Общие с предыдущими моделями пункты касаются Mistral так же и здесь не повторяются:
- **1** — падение `generate` за `max_position_embeddings`. Воспроизведено с `max_position_embeddings=16`, промптом 10 и `max_new_tokens=10`: с кэшем `RuntimeError: shape '[1, 1, 1, 4]' is invalid for input of size 0` из `RoPE.forward`. Для Mistral это особенно заметно: sliding window и rolling-buffer кэш позволяют генерировать сколь угодно долго, и мешает только таблица cos/sin.
- **3** — игнорируется `attention_mask`. `Mistral.forward` его не принимает, а `generate` принимает: `attention_mask` из нулей даёт тот же результат, что и без маски.
- **4** — нет валидации аргументов `generate`. `temperature=0.0` и `top_k=5, top_p=0.9` принимаются молча.
- **8** — `use_cache=True` по умолчанию и нет `no_grad`. В `eval()` логиты имеют `requires_grad=True`.
- **10** — интерфейс `BaseModel`.
- **12** — мёртвые проверки `hasattr(torch, "bool")` и сравнения `== True` / `!= None`.
- **18** — дублирование `generate`.
- **19** — `top_k=100` при `vocab_size=50` падает с `RuntimeError: selected index k out of range`.
- **22** — `**kwargs` в `generate`: `max_lenght=5` проглатывается без ошибки.
- **24** — bias во всех `Linear`: у Mistral 7B проекции тоже без bias. Воспроизведено: `_heads._q.bias is not None`, `_linear.bias is not None`.
- **25** — RoPE-буферы в `state_dict` по копии на слой.

### Баги

#### 27. Нет маски при кэше и `seq_len > 1` в `GroupedQueryAttention` — P1

- **Где:** `GroupedQueryAttention.forward`, `if cache is None: scores = scores.masked_fill(...)`.
- **Что:** то же, что пункт 2, но в отдельном модуле GQA, и ломается не только causal-часть, но и окно: токены куска видят будущее внутри куска, а ключи из кэша не обрезаются по окну для каждой строки. При одном новом токене маска не нужна: кэш содержит ровно `window_size` позиций, плюс сам токен — это `W + 1`, как и в маске без кэша.
- **Воспроизведено:** префилл 4 + 6 токенов через кэш расходится с полным forward на 0.26 по логитам (5 + 5 — 0.25, 6 + 4 — 0.18) при `window_size=4`.
- **Исправление:** при кэше строить маску по абсолютным позициям: строки `start_pos … start_pos + T − 1`, столбцы — позиции ключей `start_pos − len(k_cache) … start_pos + T − 1`, разрешено `0 ≤ i − j ≤ window_size`.
- **Статус:** общая часть упомянута в [известных ограничениях](README.md#известные-ограничения).

#### 28. Ключ `head_size` в конфиге игнорируется — P2

- **Где:** `Mistral.__init__`, `head_size=config["embed_dim"] // config["num_q_heads"]` для `RoPE` и `MistralDecoder`.
- **Что:** в [`mistral_train.json`](../experiments/llm_only/configs/mistral_train.json) задан `"head_size": 64`, и он совпадает с `256 // 4` случайно. Если изменить одно из значений, второе молча не подстроится.
- **Воспроизведено:** конфиг с `"head_size": 16` при `embed_dim=32, num_q_heads=4` даёт `head_size=8`.
- **Исправление:** читать `config.get("head_size", embed_dim // num_q_heads)`. Если размер задан явно, `num_q_heads * head_size` может не равняться `embed_dim` — выходная проекция `_layer` это уже поддерживает.
- **Статус:** упомянуто в [известных ограничениях](README.md#известные-ограничения) для всех моделей.

#### 29. Нет проверок `num_q_heads` и `num_kv_heads` — P2

- **Где:** `GroupedQueryAttention.__init__`, `Mistral.__init__`.
- **Что:**
  - `num_q_heads % num_kv_heads != 0` принимается конструктором и падает только в первом `forward` внутри `_repeat_kv_heads` с непонятной ошибкой `reshape`;
  - `embed_dim % num_q_heads != 0` молча усекает размер голов (как пункт 20).
- **Воспроизведено:** `num_q_heads=4, num_kv_heads=3` — `RuntimeError: shape '[1, 4, 10, 8]' is invalid for input of size 240` при `forward`. `embed_dim=32, num_q_heads=3` — Q-проекция на 30 измерений.
- **Исправление:** `ValueError` в `__init__` с понятным сообщением для обоих условий.

### Отклонения от Mistral 7B

#### 30. Размер скрытого слоя SwiGLU — P3

- **Что:** Mistral 7B использует `hidden_dim = 14336` при `dim = 4096` (3.5·d, `intermediate_size` в HF). Здесь `4·d` в каждой из трёх матриц, то есть FFN примерно на 14% тяжелее. Исправление общее с пунктом 23: параметр `hidden_dim` в `SwiGLU`, для Mistral — из конфига.

### Качество кода

#### 31. `_tril_mask` в `state_dict` — P2

- **Где:** `GroupedQueryAttention.__init__`, `register_buffer("_tril_mask", ...)`.
- **Что:** то же, что пункт 17, но в `GroupedQueryAttention`, поэтому исправление в `MultiHeadAttention` его не закроет. Маска `max_seq_len × max_seq_len` хранится в каждом слое и привязывает чекпоинт к `max_seq_len` и `window_size`.
- **Воспроизведено:** ключи `_decoders.{i}._heads._tril_mask` в `state_dict`.
- **Исправление:** `persistent=False`, либо строить маску на лету по позициям (заодно закрывает пункт 27).

#### 32. Совместимость с PyTorch < 1.2 сделана наполовину — P3

- **Где:** `GroupedQueryAttention.__init__` (`mask.bool() if hasattr(torch, "bool") else mask.byte()`), `~self._tril_mask[...]` в `forward`, top-k/top-p в `Mistral.generate`.
- **Что:** на старом torch маска становится uint8, а на современном `~` для uint8 — побитовое НЕ (`[254, 255, …]`), и индексация uint8-маской падает с `RuntimeError`. Сама fallback-ветка на старом torch не проверялась. Внешний стенд с torch < 1.2 прошла только версия на float-масках с `== 0`.
- **Исправление:** выбрать одно. Либо перейти на float-маски и `masked_fill(mask == 0, ...)` во всём коде, либо отказаться от поддержки torch < 1.2 и убрать все `hasattr(torch, "bool")` (см. пункт 12).

#### 33. Документация и мусор — P3

- Докстринг `Mistral`: название статьи выдумано («Mistral: Fast and Efficient Dense and Mixture of Experts Transformer Models»), настоящее — «Mistral 7B».
- Докстринг `GroupedQueryAttention`: ссылка «Self-attention with linear complexity (Vila et al.) arXiv:2302.05442» не соответствует статье; утверждение, что GQA используется в GPT-4, не подтверждено; обещано требование `num_q_heads * head_size == emb_size`, которое не проверяется.
- Докстринг `MistralDecoder` описывает «стек декодеров» с аргументом `num_layers`, хотя это один блок и такого аргумента нет; «RMSNorm перед и после» — на деле только pre-norm.
- Параметр `mask` в `GroupedQueryAttention.forward` и `MistralDecoder.forward` принимается и не используется.
- Закомментированный код: старый `PositionalEmbeddings` и `pos_out` в `Mistral`, старый блок кэширования и `_repeat_kv_heads` в `GroupedQueryAttention.forward`.
- Неиспользуемые переменные и импорты: `k_seq_len` в `GroupedQueryAttention.forward`; `vocab_size`, `masked_logits` в `generate`; `sqrt`, `Tensor` в `mistral.py`.
- Комментарии в `GroupedQueryAttention.forward`: сбитая нумерация шагов («Шаг 2», «3.», «5.», «8.», снова «3.», «4.») и неверные размерности (`# [B, T, hs]` там, где `[B, H, T, hs]`).
- Кэш пересобирается через `torch.cat` и срез на каждом шаге. Для учебного кода это приемлемо, но настоящего rolling buffer (запись по индексу `pos % W`) нет, хотя документация так его называет.
- Нет `save`/`load` в `Mistral`, хотя версия для внешнего стенда их требует.
- [`tests/models/test_mistral.py`](../llm/tests/models/test_mistral.py) проверяет только формы. Нет тестов на префилл кусками с кэшем (пункт 27), на генерацию до границы `max_position_embeddings` (пункт 1), на проверки из пункта 29 и на чтение `head_size` из конфига (пункт 28).

## Mixtral

Модель: [`models/mixtral/mixtral.py`](../llm/src/llm/models/mixtral/mixtral.py), блок: [`core/mixtral_decoder.py`](../llm/src/llm/core/mixtral_decoder.py), FFN: [`core/moe.py`](../llm/src/llm/core/moe.py) поверх [`core/swi_glu.py`](../llm/src/llm/core/swi_glu.py). Attention — тот же `GroupedQueryAttention`, что у Mistral.

Сама математика MoE верна: выход совпадает с наивным циклом по токенам (для каждого токена сумма `softmax(top-k логитов) · expert(x)`) с точностью 7e-8.

Общие с предыдущими моделями пункты касаются Mixtral так же и здесь не повторяются:
- **1** — падение `generate` за `max_position_embeddings`. Воспроизведено с `max_position_embeddings=16`, промптом 10 и `max_new_tokens=10`: с кэшем `RuntimeError: shape '[1, 1, 1, 4]' is invalid for input of size 0` из `RoPE.forward`.
- **3**, **4**, **22** — `attention_mask` и `**kwargs` в `generate` игнорируются, аргументы не валидируются.
- **8** — `use_cache=True` по умолчанию и нет `no_grad`. Воспроизведено: в `train()` `forward` возвращает кэш; `Trainer` вызывает `self.model(input_ids)` и собирает K/V всех слоёв на каждом шаге.
- **10**, **12**, **18**, **19** — интерфейс `BaseModel`, мёртвые `hasattr(torch, "bool")`, дублирование `generate`, пограничные случаи top-k.
- **23**, **24** — SwiGLU с `4·d` и bias во всех `Linear`. У Mixtral 8x7B эксперт — `hidden_dim = 14336` при `dim = 4096`, все проекции, включая роутер, без bias.
- **25** — RoPE-буферы в `state_dict` по копии на слой.
- **27** — нет маски при кэше и `seq_len > 1`. Воспроизведено на Mixtral: префилл 6 + 8 токенов через кэш расходится с полным forward на 0.35 по логитам при `window_size=5`.
- **28** — ключ `head_size` игнорируется. В [`mixtral_train.json`](../experiments/llm_only/configs/mixtral_train.json) `"head_size": 64` совпадает с `256 // 4` случайно.
- **29**, **31**, **32**, **33** — проверки голов, `_tril_mask` в `state_dict`, половинчатая совместимость с torch < 1.2, мусор в `GroupedQueryAttention` (неиспользуемые `mask` и `k_seq_len`).

### Баги

#### 34. MoE падает в bf16/fp16 — P1

- **Где:** `MoE.forward`, `weights_for_expert = torch.zeros(batch_size, seq_len, device=x.device)`.
- **Что:** буфер весов создаётся без `dtype` и всегда float32. Запись в него `topk_weights[...]` в bf16/fp16 падает; обучение и инференс Mixtral в половинной точности невозможны.
- **Воспроизведено:** `MoE(16, 4, 2).to(torch.bfloat16)` на bf16-входе — `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source`.
- **Исправление:** `dtype=x.dtype`, либо переписать сборку выхода без промежуточного буфера (см. пункт 39).

#### 35. Нет `save`/`load`, хотя докстринг их обещает — P2

- **Где:** докстринг `Mixtral`: «save(path)/load(path, device) — сохранение и восстановление обученной модели».
- **Что:** методов нет ни в `Mixtral`, ни в `BaseModel`.
- **Воспроизведено:** `hasattr(Mixtral, "save")`, `hasattr(Mixtral, "load")` — `False`.
- **Исправление:** реализовать в `BaseModel` (`state_dict` + `config`, `load` как `classmethod`) — закроет и Mistral, и LLaMA. Версия для внешнего стенда уже содержит рабочий вариант с полным набором аргументов конструктора.

#### 36. `top_k_experts=0` принимается — P3

- **Где:** `MoE.__init__` проверяет только `top_k_experts > num_experts`.
- **Что:** при `top_k_experts=0` ни один эксперт не выбирается, FFN-ветка тождественно возвращает нули, модель молча превращается в attention-only.
- **Воспроизведено:** `Mixtral` с `top_k_experts=0` строится и выполняет `forward` без ошибок.
- **Исправление:** `ValueError` при `top_k_experts < 1`.

### Отклонения от Mixtral 8x7B

#### 37. Нет load-balancing loss у роутера — P2

- **Где:** `MoE.forward` возвращает только выход; логиты роутера наружу не отдаются, `Trainer` считает только cross-entropy.
- **Что:** в Mixtral (как в Switch Transformer и GShard, на которые ссылается докстринг) при обучении добавляется вспомогательный loss `num_experts · Σ fᵢ · Pᵢ` (доля токенов на эксперта × средняя вероятность роутера). Без него роутер склонен схлопываться на пару экспертов, остальные не обучаются, и MoE вырождается в узкий dense FFN.
- **Исправление:** возвращать из `MoE` (или копить в атрибуте) `router_logits`, считать aux loss в модели с коэффициентом из конфига (`router_aux_loss_coef`, в HF `MixtralConfig` по умолчанию 0.001) и прибавлять в `Trainer`. Полезна и метрика загрузки экспертов в логах обучения.
- **Статус:** не задокументировано.

#### 38. Двойной dropout в MoE — P2

- **Где:** `nn.Dropout` внутри каждого `SwiGLU` и ещё один на выходе `MoE`.
- **Что:** выход эксперта прорежается дважды, и эффективная вероятность выше заданной `dropout`. В Mixtral dropout в FFN нет вовсе.
- **Воспроизведено:** при `dropout=0.5` в `train()` обнуляется 62% элементов выхода MoE вместо 50% (`0.5 + 0.5 · 0.5²` для двух экспертов).
- **Исправление:** оставить один dropout — на выходе `MoE` — и создавать экспертов с `dropout=0.0` (или добавить в `SwiGLU` флаг).

### Качество кода

#### 39. Неэффективная сборка выхода MoE — P3

- **Где:** `MoE.forward`.
- **Что:** на каждого эксперта создаётся полный буфер `[batch, seq_len]` и выполняется вложенный цикл по `top_k`, токены выбираются булевыми масками. Работает, но делает лишнюю работу и не совместимо с torch < 1.2 (булева индексация).
- **Исправление:** плоский вход `[N, emb]`, `(topk_indices == e).nonzero()` даёт пары (токен, позиция в top-k), веса — `topk_weights[token_idx, k_idx]`, выход — `output.index_add_(0, token_idx, w · expert(x[token_idx]))`. Этот вариант уже проверен во внешнем стенде и заодно закрывает пункт 34.

#### 40. Документация и мусор — P3

- В References нет самой статьи Mixtral — «Mixtral of Experts», Jiang et al., 2024, arXiv:2401.04088; есть только пост в блоге.
- Ссылка на GQA в `mixtral_decoder.py` и `mixtral.py` — `arXiv:2305.14236`, правильно `arXiv:2305.13245` (Ainslie et al.).
- Неиспользуемые импорты: `Tensor`, `F`, `sqrt` в `mixtral.py`; `F` в `mixtral_decoder.py`. Параметр `mask` в `MixtralDecoder.forward` не используется.
- Неиспользуемые переменные в `generate`: `vocab_size`, `masked_logits` лишь дублирует `logits_scaled`.
- Роутер создаётся с bias; в Mixtral `gate` — `Linear(dim, num_experts, bias=False)` (частный случай пункта 24).
- Тесты: [`test_moe.py`](../llm/tests/core/test_moe.py) проверяет формы, градиенты и детерминизм, но не корректность против эталона; нет тестов на bf16 (пункт 34), на префилл кусками с кэшем (пункт 27) и на генерацию до границы `max_position_embeddings` (пункт 1). В [`test_mixtral.py`](../llm/tests/models/test_mixtral.py) только формы.

## Gemma

Модель: [`models/gemma/gemma.py`](../llm/src/llm/models/gemma/gemma.py), блок: [`core/gemma_decoder.py`](../llm/src/llm/core/gemma_decoder.py), attention: [`core/multi_query_attention.py`](../llm/src/llm/core/multi_query_attention.py) с [`core/rope.py`](../llm/src/llm/core/rope.py), FFN: [`core/geglu.py`](../llm/src/llm/core/geglu.py). `MultiQueryAttention` — отдельный модуль, поэтому исправления в `MultiHeadAttention` и `GroupedQueryAttention` его не затрагивают.

Кэшированная генерация по одному токену совпадает с полным forward (покрыто [`test_kv_cache.py`](../llm/tests/models/test_kv_cache.py)).

Общие с предыдущими моделями пункты касаются Gemma так же и здесь не повторяются:
- **1** — падение `generate` за `max_position_embeddings`. Воспроизведено с `max_position_embeddings=16`, промптом 10 и `max_new_tokens=10`: с кэшем `RuntimeError: shape '[1, 1, 1, 4]' is invalid for input of size 0` из `RoPE.forward`, без кэша `ValueError`. `Gemma.forward` пропускает проверку длины при кэше, а `MultiQueryAttention` сравнивает с лимитом только `seq_len`, без `start_pos`.
- **3** — игнорируется `attention_mask`. `Gemma.forward` его не принимает, `generate` принимает: `attention_mask` из нулей даёт тот же результат, что и без маски. Параметр `mask` в `GemmaDecoder.forward` и `MultiQueryAttention.forward` тоже не используется.
- **4** — нет валидации аргументов `generate`. `temperature=0.0` и `top_k=5, top_p=0.9` принимаются молча.
- **8** — `use_cache=True` по умолчанию и нет `no_grad`. В `eval()` логиты имеют `requires_grad=True`, кэш возвращается по умолчанию.
- **10**, **12**, **18** — интерфейс `BaseModel`, мёртвые `hasattr(torch, "bool")` и сравнения `== True` / `!= None`, дублирование `generate`.
- **19** — `top_k=100` при `vocab_size=50` падает с `RuntimeError: selected index k out of range`.
- **20** — проверка делимости. Воспроизведено: `embed_dim=34, num_q_heads=4` принимается, Q-проекция на 32 измерения. При нечётном `head_size` — `assert` в `RoPE.__init__` без упоминания `embed_dim` и `num_q_heads`.
- **22** — `**kwargs` в `generate`: `max_lenght=5` проглатывается без ошибки.
- **25** — RoPE-буферы в `state_dict` по копии на слой. Воспроизведено: при `num_layers=2` три ключа `*.cos_matrix`.
- **28** — ключ `head_size` игнорируется. Воспроизведено: `"head_size": 16` при `embed_dim=32, num_q_heads=4` даёт `head_size=8`. См. также [неиспользуемые ключи конфига](gemma.md#неиспользуемые-ключи-конфига).
- **32** — половинчатая совместимость с torch < 1.2 в top-k/top-p `generate` и в `_tril_mask`. Версия Gemma на float-масках с `== 0` прошла внешний стенд 2026-09-28.
- **35** — докстринг `Gemma` обещает `save(path)/load(path, device)`, методов нет. Воспроизведено: `hasattr(Gemma, "save")` — `False`.

### Баги

#### 41. Нет causal-маски при кэше и `seq_len > 1` в `MultiQueryAttention` — P1

- **Где:** `MultiQueryAttention.forward`, `if cache is None: scores = scores.masked_fill(...)`.
- **Что:** то же, что пункты 2 и 27, но в третьем модуле attention. Токены куска, поданного вместе с кэшем, видят будущее внутри куска. `generate` подаёт по одному токену, поэтому там не проявляется.
- **Воспроизведено:** префилл 4 + 6 токенов через кэш расходится с полным forward на 0.14 по логитам.
- **Исправление:** всегда накладывать маску со сдвигом `self._tril_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]` и проверять `start_pos + seq_len <= max_seq_len` (закрывает часть пункта 1). Проверено в версии для внешнего стенда: префилл кусками совпадает с полным forward.

### Отклонения от Gemma

Сравнение с Gemma 2B/7B ([Gemma Team, 2024](https://arxiv.org/abs/2403.08295); `GemmaConfig`/`GemmaModel` в HF).

#### 42. Эмбеддинги не масштабируются на √d — P2

- **Что:** в Gemma выход `embed_tokens` умножается на `sqrt(hidden_size)` (`normalizer` в `GemmaModel.forward`) перед первым блоком. Здесь эмбеддинги идут в декодер как есть.
- **Последствия:** при tied embeddings (пункт 43) без масштабирования вход в первый блок на порядок меньше по норме, чем предполагает архитектура. Веса Gemma дают неверный результат даже при совпадении остальных слоёв.
- **Исправление:** `out = tok_out * math.sqrt(embed_dim)` в `Gemma.forward` (в HF константа приводится к dtype эмбеддингов).

#### 43. Нет weight tying, bias во всех `Linear` — P2

- **Что:** в Gemma выходная проекция привязана к `embed_tokens` (`tie_word_embeddings=True`), и все проекции без bias (`attention_bias=False`). Здесь `_linear` — отдельный `nn.Linear` с bias, bias есть в Q/K/V, выходной проекции attention и трёх матрицах GeGLU.
- **Воспроизведено:** `m._linear.bias is not None`, `m._decoders[0]._heads._q.bias is not None`, `m._linear.weight is not m._token_embeddings._embedding.weight`.
- **Последствия:** у Gemma словарь 256 000 токенов, поэтому отдельная голова — это лишние ~524M параметров для 2B (`256000 × 2048`), то есть около четверти модели.
- **Исправление:** как в пунктах 5 и 24: `bias=False` под флагом конфига, `_linear.weight = _token_embeddings._embedding.weight`.

#### 44. GeGLU с hidden = 4·d вместо 8·d — P2

- **Где:** [`core/geglu.py`](../llm/src/llm/core/geglu.py), `nn.Linear(emb_size, 4 * emb_size)` для `_gate`, `_up`, `_down`.
- **Что:** в Gemma `intermediate_size` = 16384 при `hidden_size` = 2048 (2B) и 24576 при 3072 (7B), то есть 8·d. Здесь 4·d — FFN вдвое уже, чем в статье. В отличие от пункта 23 (LLaMA), здесь FFN не тяжелее, а легче оригинала.
- **Исправление:** параметр `hidden_dim` в `GeGLU` с чтением из конфига, как предложено для `SwiGLU` в пункте 23.

#### 45. Нельзя выразить Gemma 7B: MQA всегда, `head_size` = d / heads — P2

- **Что:** MQA (одна K/V-голова) используется только в Gemma 2B. Gemma 7B — обычный MHA с 16 головами, и `head_dim = 256` не равен `hidden_size / num_heads` (16 × 256 = 4096 ≠ 3072). Здесь `MultiQueryAttention` всегда с одной K/V-головой, а `head_size` всегда `embed_dim // num_q_heads` (пункт 28).
- **Исправление:** заменить `MultiQueryAttention` на `GroupedQueryAttention` с `num_kv_heads` из конфига (MQA — частный случай `num_kv_heads=1`) и читать `head_size` из конфига. `_layer` уже умеет проецировать `num_q_heads * head_size ≠ embed_dim` обратно в `embed_dim`. Тогда же уйдёт отдельный модуль MQA и пункты 41 и 47 закроются вместе с 27 и 31.

#### 46. RMSNorm без `(1 + w)` и вычислений во float32 — P3

- **Где:** [`core/rms_norm.py`](../llm/src/llm/core/rms_norm.py).
- **Что:** в Gemma `GemmaRMSNorm` хранит вес, инициализированный нулями, и умножает на `(1 + weight)`, а нормализацию считает во float32 и приводит результат обратно. Здесь вес инициализирован единицами и умножается напрямую, вычисления в dtype входа. При обучении с нуля параметризации эквивалентны, но веса Gemma без поправки `+1` не загрузятся корректно, а в bf16 нормализация менее точна.
- **Исправление:** для загрузки весов — прибавлять 1 при конвертации. Для bf16 — `x.float()` внутри `forward` и `.to(x.dtype)` на выходе (затрагивает LLaMA, Mistral, Mixtral).
- **Замечание по загрузке весов HF в целом:** помимо пунктов 42–46 нужна перестановка строк `q_proj`/`k_proj` — HF Gemma использует `rotate_half`, здесь чередующиеся пары (как в пункте 24).

### Качество кода

#### 47. `_tril_mask` в `state_dict` — P2

- **Где:** `MultiQueryAttention.__init__`, `register_buffer("_tril_mask", ...)`.
- **Что:** то же, что пункты 17 и 31, но в `MultiQueryAttention`, поэтому их исправление его не закроет.
- **Воспроизведено:** ключи `_decoders.{i}._heads._tril_mask` в `state_dict`.
- **Исправление:** `persistent=False`.

#### 48. Документация и мусор — P3

- Докстринги `Gemma` и `GemmaDecoder` описывают несуществующие варианты: «Multi-Query либо Grouped heads», «FFN с GeGLU/SwiGLU», «RMSNorm или LayerNorm»; псевдокод в `GemmaDecoder` использует `LayerNorm`. В коде всегда MQA + GeGLU + RMSNorm.
- Неверная ссылка на статью Gemma в докстрингах `Gemma` и `Gemma.generate`: `arXiv:2403.07794`, правильно `arXiv:2403.08295`.
- Неиспользуемые импорты: `math`, `sqrt`, `Tensor` в `gemma.py`; `F` в `gemma_decoder.py`. Неиспользуемая переменная `vocab_size` в `generate`, `masked_logits` лишь дублирует `logits_scaled`.
- Комментарии в `MultiQueryAttention.forward`: сбитая нумерация шагов («Шаг 2», «3.», «5.», снова «3.», «4.») и неверные размерности (`# [B, T, hs]` там, где `[B, H, T, hs]`).
- [`gemma_train.json`](../experiments/llm_only/configs/gemma_train.json) и [`gemma_generate.json`](../experiments/llm_only/configs/gemma_generate.json) содержат ключи Mixtral (`num_kv_heads`, `num_experts`, `top_k_experts`, `window_size`), которые модель не читает. Задокументировано в [gemma.md](gemma.md#неиспользуемые-ключи-конфига), но проще убрать их из JSON.
- Тесты: [`test_gemma.py`](../llm/tests/models/test_gemma.py) проверяет только формы. `test_forward_masked` в [`test_gemma_decoder.py`](../llm/tests/core/test_gemma_decoder.py) передаёт маску и проверяет лишь форму, создавая впечатление, что маска поддерживается (пункт 3). Нет тестов на префилл кусками с кэшем (пункт 41) и на генерацию до границы `max_position_embeddings` (пункт 1).
