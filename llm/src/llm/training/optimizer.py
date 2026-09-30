r"""
Модуль оптимизации для обучения нейронных сетей.

В данном модуле реализована функция выбора и инициализации оптимизаторов, наиболее популярных при обучении глубоких нейросетей:
- AdamW
- Adam
- SGD

Теоретическое обоснование:
--------------------------
Задача оптимизации в обучении нейросети заключается в минимизации функции потерь (Loss) по параметрам модели W. Современные методы базируются на стохастическом градиентном спуске (SGD), а также на его адаптивных модификациях (Adam, AdamW).

**SGD** (Stochastic Gradient Descent) — стохастический градиентный спуск:
  W_{t+1} = W_t - \eta \nabla_W L(W_t)
  Здесь \eta — шаг обучения, \nabla_W — градиент по параметрам. SGD позволяет случайно выбирать подмножество обучающих данных для каждой итерации, что ускоряет процесс и уменьшает избыточную корреляцию между примерами.

**Adam** (Adaptive Moment Estimation) — адаптивный алгоритм, который использует скользящую среднюю не только градиентов, но и их квадратов:
  m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_W L(W_t)
  v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_W L(W_t))^2
  W_{t+1} = W_t - \eta m_t/(\sqrt{v_t}+\epsilon)
  Где \beta_1, \beta_2 — коэффициенты экспоненциального сглаживания.

**AdamW** — модификация Adam, в которой weight decay (имплицитная L2-регуляризация) вводится корректно, отдельно от шага градиента, что улучшает обобщающую способность моделей:
  W_{t+1} = W_t - \eta [ m_t/(\sqrt{v_t}+\epsilon) + \lambda W_t ]
  Где \lambda — коэффициент weight decay.

Детальное описание: https://arxiv.org/abs/1711.05101

Weight decay применяется только к матрицам — весам Linear и эмбеддингам (параметры с dim ≥ 2).
Смещения и коэффициенты нормализации (LayerNorm, RMSNorm) не затухают: так в GPT-1 (разд. 4.1,
"all non bias or gain weights"), nanoGPT и HF Trainer. Их мало, на переобучение они почти не
влияют, а затухание коэффициента нормализации к нулю лишь уменьшает масштаб сигнала.

Пример использования:
---------------------
>>> optimizer = get_optimizer(model, lr=3e-4, weight_decay=0.01, optimizer_type="adamw")
>>> for batch in dataloader:
...     loss = model(batch)
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()

"""
import torch.optim as optim


def weight_decay_param_groups(model, weight_decay):
    """
    Делит параметры модели на две группы для оптимизатора: матрицы (dim ≥ 2: веса Linear,
    эмбеддинги) — с weight_decay, остальное (bias, веса LayerNorm и RMSNorm) — без.

    Общая матрица при weight tying входит один раз: model.parameters() не повторяет
    параметр, зарегистрированный в двух модулях. Пустая группа не создаётся.

    Returns:
        list[dict] — группы параметров для конструктора torch.optim.Optimizer.
    """
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return [group for group in groups if group["params"]]


def get_optimizer(model, lr=3e-4, weight_decay=0.01, optimizer_type="adamw"):
    """
    Фабричная функция для создания оптимизатора PyTorch по выбранному типу.
    
    Параметры
    ---------
    model : torch.nn.Module
        Модель, параметры которой требуется оптимизировать.
    lr : float, по умолчанию 3e-4
        Шаг обучения (learning rate).
    weight_decay : float, по умолчанию 0.01
        Коэффициент weight decay. Применяется только к матрицам (веса Linear и эмбеддинги),
        bias и веса нормализаций не затухают (см. weight_decay_param_groups).
    optimizer_type : str, по умолчанию 'adamw'
        Тип оптимизатора:
        - 'adamw' — AdamW: decoupled weight decay, θ ← θ − η·λ·θ отдельно от шага Adam;
        - 'adam' — Adam с L2-регуляризацией: λ·θ прибавляется к градиенту и нормируется
          вместе с ним, поэтому на весах с большими градиентами почти не действует;
        - 'sgd' — SGD с моментом 0.9; weight decay — тоже L2 через градиент.
    
    Возвращаемое значение
    ---------------------
    torch.optim.Optimizer
        Объект-оптимизатор, готовый к использованию.

    Исключения
    ----------
    ValueError: Если передан неизвестный тип оптимизатора.

    Пример использования:
    ---------------------
    >>> optimizer = get_optimizer(model, lr=1e-3, optimizer_type='sgd')
    """
    optimizer_classes = {"adamw": optim.AdamW, "adam": optim.Adam, "sgd": optim.SGD}
    optimizer_class = optimizer_classes.get(optimizer_type.lower())
    if optimizer_class is None:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
    groups = weight_decay_param_groups(model, weight_decay)
    kwargs = {"momentum": 0.9} if optimizer_class is optim.SGD else {}
    return optimizer_class(groups, lr=lr, weight_decay=weight_decay, **kwargs)
