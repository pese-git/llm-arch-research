import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, head_size: int, max_seq_len: int, base: int = 10_000):
        super().__init__()
        assert head_size % 2 == 0, "head_size должен быть четным"

        # Обратные частоты
        freqs = 1.0 / (base ** (2 * torch.arange(head_size // 2).float() / head_size))
        
        # Позиции
        positions = torch.arange(max_seq_len).float()
        
        # Матрица частот (внешнее произведение)
        #freq_matrix = torch.outer(positions, freqs)
        freq_matrix = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Матрицы косинусов и синусов
        self.register_buffer('cos_matrix', torch.cos(freq_matrix))
        self.register_buffer('sin_matrix', torch.sin(freq_matrix))


    def forward(self, x: torch.Tensor): # Получает на вход тензор x (тип float) размером [batch_size × seq_len × head_size]
        seq_len = x.size(1)
        # Берем нужную часть матриц и приводим к типу x
        cos = self.cos_matrix[:seq_len].to(x.dtype)  # [seq_len, head_size//2]
        sin = self.sin_matrix[:seq_len].to(x.dtype)  # [seq_len, head_size//2]
        

        # Разделяем на четные и нечетные
        x_even = x[:, :, 0::2]  # [batch_size, seq_len, head_size//2]
        x_odd = x[:, :, 1::2]   # [batch_size, seq_len, head_size//2]

        # Применяем поворот
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos


        # Объединяем обратно
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # [batch_size, seq_len, head_size]

        return x_rotated