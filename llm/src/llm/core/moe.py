import torch
from torch import nn
import torch.nn.functional as F
from llm.core.swi_glu import SwiGLU

class MoE(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_experts: int,
        top_k_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._num_experts = num_experts
        self._top_k_experts = top_k_experts

        self._router = nn.Linear(emb_size, num_experts)
        self._experts = nn.ModuleList([SwiGLU(
            emb_size=emb_size,
            dropout=dropout,
        ) for _ in range(num_experts)])
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
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