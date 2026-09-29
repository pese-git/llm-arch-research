import torch
from torch import nn
import torch.nn.functional as F
from llm.core.swi_glu import SwiGLU

class MoE(nn.Module):
    """
    MoE (Mixture of Experts) — слой «смеси экспертов» для современных трансформерных архитектур с разреженной активацией.

    Назначение:
    -----------
    Класс реализует слой разреженного условного вычисления для увеличения capacity трансформеров без роста вычислительных затрат.
    Для каждого токена из последовательности выбирается (с помощью роутера) наиболее подходящее подмножество экспертов (малых нейросетей).
    Итоговый выход формируется как взвешенная сумма откликов экспертов, выбранных для данного токена.

    Архитектурная схема:
    ---------------------
    - Для каждого входного токена `x` роутер (обычно один Linear-слой) предсказывает skor, насколько каждый из `num_experts` релевантен.
    - Для каждого токена выбираются top_k_experts с максимальными skor; только они обрабатывают этот токен.
    - Каждый эксперт здесь представлен отдельным экземпляром блока `SwiGLU` (может быть любая небольшая feed-forward сеть).
    - Выход каждого эксперта умножается на вес (softmax по top-K логитам роутера) — агрегируется взвешенная сумма.
    - Dropout применяется к итоговому выходу.

    Математика (коротко):
    ---------------------
        Пусть X ∈ R^{BxSxD} — вход, 
        E — число экспертов,
        K — число активируемых экспертов на токен.
        l(x) = W_r x — логиты роутера; берутся top-K логитов и их индексы,
        веса w = softmax(top-K логитов) — нормированы только по выбранным экспертам.
        Для каждого токена:
            y_j = Expert_j(x)
            y = sum_j(w_j * y_j), где j пробегает по выбранным экспертам
        Output: Y ∈ R^{BxSxD}

    Аргументы конструктора:
    ----------------------
    emb_size : int
        Размерность входных/выходных векторов (обычно совпадает с embedding модели).
    num_experts : int
        Общее число экспертов внутри слоя MoE.
    top_k_experts : int
        Сколько экспертов активировать и агрегировать на каждом токене (обычно 2-8).
    dropout : float, по умолчанию 0.1
        Dropout к выходу агрегатора.

    Пример использования:
    ---------------------
        >>> moe = MoE(emb_size=512, num_experts=8, top_k_experts=2, dropout=0.1)
        >>> x = torch.randn(4, 16, 512)
        >>> y = moe(x)
        >>> y.shape    # torch.Size([4, 16, 512])

    Литература:
    -----------
    - Shazeer, N. et al. “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, 2017. https://arxiv.org/abs/1701.06538
    - Fedus, W., Zoph, B., & Shazeer, N. “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity”, 2021. https://arxiv.org/abs/2101.03961
    - Mistral/Mixtral: https://mistral.ai/news/mixtral-of-experts/
    """
    def __init__(
        self,
        emb_size: int,
        num_experts: int,
        top_k_experts: int,
        dropout: float = 0.1,
    ):
        """
        Конструктор слоя MoE (Mixture of Experts).

        Позволяет создать слой, состоящий из набора экспертов (например, отдельных небольших feedforward-нейросетей) и роутера,
        который будет для каждого токена определять наиболее релевантных экспертов.
        Часть экспертов (top_k_experts) активируется для каждого токена, остальные — пропускаются.

        Аргументы:
        ----------
        emb_size : int
            Размерность входных и выходных векторов (embedding size).
            Определяет, над каким пространством признаков будет работать роутер и эксперты.
            Например, если скрытый размер слоя трансформера 512, сюда нужно передать 512.

        num_experts : int
            Общее количество экспертов в слое MoE.
            Чем больше экспертов — тем больше capacity у модели, но тем выше требования к RAM/VRAM при обучении.
            Пример: 8, 16, 32, 64.

        top_k_experts : int
            Сколько экспертов одновременно будет обрабатывать каждый токен.
            Обычно 2–8. Меньшее значение — выше разреженность, больше экономия вычислений.

        dropout : float, по умолчанию 0.1
            Вероятность зануления значений на выходе после агрегации откликов экспертов.
            Используется для регуляризации (борьбы с переобучением). Это единственный dropout
            слоя: эксперты создаются без собственного.

        Пример:
        -------
            >>> moe = MoE(emb_size=256, num_experts=8, top_k_experts=2, dropout=0.1)
            >>> print(moe)
            MoE( ... )

        Теория:
        -------
            Слой строит:
            - Линейный роутер (Linear(emb_size, num_experts)): выдает «важность» каждого эксперта для токена.
            - Список из num_experts экспертов (в данной реализации — SwiGLU-блоки).
            
            При каждом проходе для каждого токена выбираются top_k_experts наиболее релевантных экспертов,
            их ответы агрегируются взвешенной суммой (softmax по роутерным логитам).
        """
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts должно быть ≥ 1, получено {num_experts}")
        if not 1 <= top_k_experts <= num_experts:
            # При top_k_experts=0 не выбирается ни один эксперт, и MoE молча возвращает нули
            raise ValueError(
                f"top_k_experts ({top_k_experts}) должен быть от 1 до num_experts ({num_experts})"
            )
        self._num_experts = num_experts
        self._top_k_experts = top_k_experts

        self._router = nn.Linear(emb_size, num_experts)
        # Эксперты без собственного dropout: он один — на выходе MoE. Иначе выход эксперта
        # прорежался бы дважды, и эффективная вероятность была бы выше заданной
        self._experts = nn.ModuleList([SwiGLU(
            emb_size=emb_size,
            dropout=0.0,
        ) for _ in range(num_experts)])
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход (forward) через слой MoE.

        Для входной последовательности скрытых состояний (обычно из предыдущего слоя трансформера)
        данный метод динамически выбирает для каждого токена топ-k наиболее релевантных экспертов с помощью роутера,
        пропускает соответствующие токены через выбранных экспертов и агрегирует их результаты.

        Математически:
        --------------
          1. Для каждого токена вычисляются логиты маршрутизатора (роутера):  
               router_logits = Linear(x) ∈ ℝ^{batch, seq, num_experts}
          2. Выбираются top_k экспертов (topk_indices) и соответствующие им softmax-веса (topk_weights).
          3. Каждый эксперт обрабатывает только свой поднабор токенов.
          4. Результат агрегируется — отклик эксперта умножается на вес, ответы суммируются для каждого токена.
          5. На результат применяется dropout для регуляризации.

        Аргументы:
        ----------
        x : torch.Tensor
            Трёхмерный входной тензор формы [batch_size, seq_length, emb_size],
            где batch_size — размер батча, seq_length — длина последовательности, emb_size — размерность эмбеддинга.

        Возвращает:
        -----------
        torch.Tensor :
            Тензор той же формы [batch_size, seq_length, emb_size] — результат комбинирования выходов выбранных экспертов
            с учетом softmax-весов маршрутизатора и dropout'а.

        Пример:
        -------
            >>> y = moe(x)
            >>> print(y.shape)
            torch.Size([batch_size, seq_length, emb_size])

        Примечание:
        -----------
        - Каждый токен чаще всего активирует только подмножество экспертов. 
        - Остальные эксперты вычислительно “спят”, что позволяет строить очень большие (по параметрам) модели с малым ростом затрат.
        - Работа с распределением топ-к экспертов и агрегирование с весами реализовано автоматически.

        """
        batch_size, seq_len, emb_size = x.shape
        # Токены батча обрабатываются одинаково, поэтому удобнее плоский вид [N, emb_size]
        x_flat = x.reshape(-1, emb_size)  # [N, emb_size], N = batch_size * seq_len

        # 1. Логиты роутера и top-k экспертов для каждого токена
        router_logits = self._router(x_flat)  # [N, num_experts]
        topk_logits, topk_indices = torch.topk(
            router_logits, k=self._top_k_experts, dim=-1
        )  # [N, top_k]

        # 2. Веса выбранных экспертов: softmax только по top-k логитам. Явно во float32 с
        # приведением к dtype входа — как в HF и эталонном коде Mistral. Встроенный softmax
        # PyTorch и так накапливает во float32, поэтому на CPU/MPS результат не меняется,
        # но точность весов не зависит от реализации softmax на конкретном backend
        topk_weights = F.softmax(topk_logits.float(), dim=-1).to(x.dtype)  # [N, top_k]

        # 3. Каждый эксперт обрабатывает только свои токены, результат с весом
        # добавляется в строки этих токенов
        output = torch.zeros_like(x_flat)  # [N, emb_size]
        for expert_id in range(self._num_experts):
            # Пары (токен, позиция в top-k), где выбран этот эксперт; в top-k эксперты
            # не повторяются, поэтому каждый токен встречается не больше одного раза
            token_idx, k_idx = torch.where(topk_indices == expert_id)
            if token_idx.numel() == 0:
                continue  # эксперт никем не выбран — не считается вовсе

            # SwiGLU ждёт [batch, seq, emb]: выбранные токены — одна «последовательность»
            expert_output = self._experts[expert_id](
                x_flat[token_idx].unsqueeze(0)
            ).squeeze(0)  # [n_selected, emb_size]
            weights = topk_weights[token_idx, k_idx].unsqueeze(-1)  # [n_selected, 1]
            output.index_add_(0, token_idx, weights * expert_output)

        out = self._dropout(output.reshape(batch_size, seq_len, emb_size))

        return out