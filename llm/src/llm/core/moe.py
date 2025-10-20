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
    - Выход каждого эксперта умножается на индивидуальный вес (softmax по skor) — агрегируется взвешенная сумма.
    - Dropout применяется к итоговому выходу.

    Математика (коротко):
    ---------------------
        Пусть X ∈ R^{BxSxD} — вход, 
        E — число экспертов,
        K — число активируемых экспертов на токен.
        r(x) = softmax(W_r x) — роутинг-логиты, top-K берём индексы и веса.
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
            Используется для регуляризации (борьбы с переобучением).

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
        if top_k_experts > num_experts:
            raise ValueError(f"top_k_experts ({top_k_experts}) должен быть меньше или равен num_experts ({num_experts})!")
        self._num_experts = num_experts
        self._top_k_experts = top_k_experts

        self._router = nn.Linear(emb_size, num_experts)
        self._experts = nn.ModuleList([SwiGLU(
            emb_size=emb_size,
            dropout=dropout,
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
        
        # 1. Пропускаем через роутер
        router_logits = self._router(x)  # [batch_size, seq_len, num_experts]
        
        # 2. Отбираем топ-k экспертов для каждого токена
        topk_logits, topk_indices = torch.topk(
            router_logits, 
            k=self._top_k_experts, 
            dim=-1
        )  # topk_logits: [batch_size, seq_len, top_k]
           # topk_indices: [batch_size, seq_len, top_k]
        
        # 3. Получаем веса через softmax и нормируем
        topk_weights = F.softmax(topk_logits, dim=-1)  # [batch_size, seq_len, top_k]
        
        # 4. Создаём нулевой тензор для результата
        output = torch.zeros_like(x)  # [batch_size, seq_len, emb_size]   

        # 5. Проходим по всем экспертам
        for expert_id in range(self._num_experts):
            # Шаг 1: Создаём маску - где находится текущий эксперт в топ-k
            expert_mask = (topk_indices == expert_id)  # [batch_size, seq_len, top_k]
            # Шаг 2: Проверяем, выбран ли эксперт хотя бы одним токеном
            if not expert_mask.any():
                continue  # Эксперт никем не выбран, переходим к следующему

            # Шаг 3: Находим токены, которые выбрали этого эксперта
            # (хотя бы в одной из top_k позиций)
            token_mask = expert_mask.any(dim=-1)  # [batch_size, seq_len]

            # Шаг 4: Отбираем токены из x
            # Отбираем токены для этого эксперта
            expert_input = x[token_mask]

            # Пропускаем через эксперта
            # Добавляем batch dimension для SwiGLU и затем убираем
            expert_output = self._experts[expert_id](
                expert_input.unsqueeze(0)
            ).squeeze(0)

            # Получаем веса для этого эксперта
            # Для каждого токена может быть несколько весов (если эксперт в топ-k несколько раз)
            # Но на практике каждый эксперт появляется максимум 1 раз в топ-k
            # Находим веса: где expert_mask == True, берём соответствующий вес
            weights_for_expert = torch.zeros(
                batch_size, seq_len, device=x.device
            )

            # Для каждой позиции в топ-k
            for k in range(self._top_k_experts):
                mask_k = topk_indices[:, :, k] == expert_id
                weights_for_expert[mask_k] = topk_weights[:, :, k][mask_k]

            # Отбираем только веса для выбранных токенов
            selected_weights = weights_for_expert[token_mask]  # [num_selected_tokens]


            # Перемножьте выход эксперта на веса текущего эксперта.
            weighted_output = selected_weights.unsqueeze(-1) * expert_output

            # Помещаем результат на своё место в выходном тензоре
            output[token_mask] += weighted_output
    
        out = self._dropout(output)

        return out