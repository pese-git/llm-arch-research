"""
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
        Коэффициент weight decay (L2-регуляризации).
    optimizer_type : str, по умолчанию 'adamw'
        Тип оптимизатора: 'adamw', 'adam' или 'sgd'.
    
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
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
