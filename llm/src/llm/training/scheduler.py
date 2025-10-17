"""
Модуль для управления динамикой шага обучения (learning rate scheduling) при обучении нейронных сетей.

Теоретическое обоснование:
--------------------------
Плавная динамика шага обучения существенно влияет на сходимость и итоговое качество моделей. Введение этапа "разогрева" (warmup) — техники, при которой шаг обучения начинается с нуля и постепенно увеличивается до целевого значения, снижает вероятность неустойчивых градиентов на старте обучения. Подобная стратегия показала свою эффективность для крупных нейронных сетей, особенно в трансформерах (Vaswani et al, 2017, https://arxiv.org/abs/1706.03762).

Линейный scheduler с warmup задаёт динамику learning rate по формуле:
  - если current_step < num_warmup_steps:
        lr = lr_init * (current_step / num_warmup_steps)
  - иначе:
        lr = lr_init * max(0, (num_training_steps - current_step) / (num_training_steps - num_warmup_steps))

Пример использования:
---------------------
>>> optimizer = get_optimizer(model, lr=3e-4)
>>> scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=10000)
>>> for step in range(num_training_steps):
...     optimizer.step()
...     scheduler.step()
"""

from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Создаёт линейный планировщик изменения шага обучения (learning rate) с этапом warmup для оптимизатора PyTorch.

    Аргументы
    ---------
    optimizer : torch.optim.Optimizer
        Оптимизатор, для которого применяется scheduler.
    num_warmup_steps : int
        Количество шагов разогрева (warmup) — начиная с нулевого шага и плавного увеличения lr до номинального значения.
    num_training_steps : int
        Общее количество шагов (эпох/итераций) обучения модели.

    Возвращаемое значение
    ---------------------
    torch.optim.lr_scheduler.LambdaLR
        Планировщик lr, который следует вызывать после каждого optimizer.step() во время обучения.

    Теоретическая справка
    ---------------------
    Такой scheduler позволяет повысить стабильность и устойчивость обучения крупных моделей (особенно трансформеров), предотвращая резкие скачки градиентов в начале.
    
    Пример:
    -------
    >>> optimizer = get_optimizer(model, lr=3e-4)
    >>> scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=10000)
    >>> for step in range(num_training_steps):
    ...     optimizer.step()
    ...     scheduler.step()
    """

    def lr_lambda(current_step):
        # Линейный рост lr на этапе разогрева
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Линейное затухание lr после разогрева
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)
