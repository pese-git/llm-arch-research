"""
Rotary Positional Embeddings (RoPE) - ротационные позиционные эмбеддинги.

Реализация ротационного позиционного кодирования, которое кодирует позиционную
информацию через вращение векторов запросов и ключей в комплексном пространстве.

Научная статья: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864

Математическая основа:
Для позиции m и измерения i:
θ_i = base^(-2i/d)
q'_m = q_m * cos(mθ_i) + rotate(q_m) * sin(mθ_i)

Преимущества:
- Относительное позиционное кодирование
- Лучшая экстраполяция на длинные последовательности
- Сохранение нормы векторов
"""

import torch
from torch import nn
from typing import Optional


class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) для механизма внимания.
    
    Кодирует позиционную информацию через вращение векторов запросов и ключей
    в многомерном пространстве с использованием синусов и косинусов.
    
    Args:
        head_size: Размерность головы внимания (должен быть четным)
        max_seq_len: Максимальная длина последовательности
        base: Базовое значение для вычисления частот (по умолчанию 10000)
    
    Attributes:
        cos_matrix: Буферизованная матрица косинусов формы [max_seq_len, head_size//2]
        sin_matrix: Буферизованная матрица синусов формы [max_seq_len, head_size//2]
    """
    
    def __init__(self, head_size: int, max_seq_len: int, base: int = 10_000):
        """
        Инициализация RoPE эмбеддингов.
        
        Args:
            head_size: Размерность головы внимания (должен быть четным)
            max_seq_len: Максимальная поддерживаемая длина последовательности
            base: Базовое значение для вычисления частот (типично 10000)
            
        Raises:
            AssertionError: Если head_size не четный
        """
        super().__init__()
        assert head_size % 2 == 0, "head_size должен быть четным"
        
        # Вычисление частот: θ_i = base^(-2i/d) для i ∈ [0, d/2-1]
        freqs = 1.0 / (base ** (2 * torch.arange(head_size // 2).float() / head_size))
        
        # Позиции от 0 до max_seq_len-1
        positions = torch.arange(max_seq_len).float()
        
        # Внешнее произведение: m * θ_i для всех позиций и частот
        freq_matrix = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Предвычисление матриц косинусов и синусов
        self.register_buffer('cos_matrix', torch.cos(freq_matrix))
        self.register_buffer('sin_matrix', torch.sin(freq_matrix))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Применение ротационного позиционного кодирования к входному тензору.
        
        Args:
            x: Входной тензор формы [batch_size, seq_len, head_size]
            
        Returns:
            Тензор с примененным RoPE формы [batch_size, seq_len, head_size]
            
        Алгоритм:
        1. Разделение векторов на четные и нечетные компоненты
        2. Применение вращения через синусы и косинусы
        3. Объединение компонент обратно
        """
        seq_len = x.size(1)
        
        # Берем нужную часть матриц и приводим к типу x
        cos = self.cos_matrix[:seq_len].to(x.dtype)  # [seq_len, head_size//2]
        sin = self.sin_matrix[:seq_len].to(x.dtype)  # [seq_len, head_size//2]
        
        # Разделяем на четные и нечетные компоненты
        x_even = x[:, :, 0::2]  # [batch_size, seq_len, head_size//2]
        x_odd = x[:, :, 1::2]   # [batch_size, seq_len, head_size//2]

        # Применяем поворот: q' = q * cos(mθ) + rotate(q) * sin(mθ)
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Объединяем обратно в исходную размерность
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # [batch_size, seq_len, head_size]

        return x_rotated