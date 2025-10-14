import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from math import sqrt
from llm.core.base_model import BaseModel


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor): # [batch_size × seq_len × emb_size]
        return torch.sigmoid(x) * x
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self._eps = eps
        self._w = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor): # [batch_size × seq_len × emb_size]
        rms = (x.pow(2).mean(-1, keepdim=True) + self._eps) ** 0.5
        norm_x = x / rms
        return self._w * norm_x

class SwiGLU(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()

        self._gate = nn.Linear(emb_size, 4 * emb_size)
        self._up = nn.Linear(emb_size, 4 * emb_size)
        self._down = nn.Linear(4 * emb_size, emb_size)
        self._activation = SiLU()
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor): # [batch_size × seq_len × emb_size].
        gate_out = self._gate(x)                          # [batch, seq, 4*emb]
        activation_out = self._activation(gate_out)       # [batch, seq, 4*emb]
        up_out = self._up(x)                              # [batch, seq, 4*emb]
        out = up_out * activation_out                     # поэлементное!
        out = self._down(out)                             # [batch, seq, emb]
        return self._dropout(out)


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._embedding(x)

    @property
    def num_embeddings(self) -> int:
        return self._embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0) / math.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            self.sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
        ))

    
    
import torch
from torch import nn
from typing import Optional


class RoPE(nn.Module):

    def __init__(self, head_size: int, max_seq_len: int, base: int = 10_000):
        super().__init__()
        assert head_size % 2 == 0, "head_size должен быть четным"

        # Вычисление частот: θ_i = base^(-2i/d) для i ∈ [0, d/2-1]
        freqs = 1.0 / (base ** (2 * torch.arange(head_size // 2).float() / head_size))

        # Позиции от 0 до max_seq_len-1
        positions = torch.arange(max_seq_len).float()

        # Внешнее произведение: m * θ_i для всех позиций и частот
        freq_matrix = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Предвычисление матриц косинусов и синусов
        self.register_buffer("cos_matrix", torch.cos(freq_matrix))
        self.register_buffer("sin_matrix", torch.sin(freq_matrix))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor: # [batch_size × seq_len × head_size] [batch_size × num_heads × seq_len × head_size]
        batch_size, num_heads, seq_len, head_size = x.shape

        # Берем нужную часть матриц и приводим к типу x
        cos = self.cos_matrix[start_pos:start_pos+seq_len].to(x.dtype)  # [seq_len, head_size//2]
        sin = self.sin_matrix[start_pos:start_pos+seq_len].to(x.dtype)  # [seq_len, head_size//2]

        # Явное изменение формы для broadcasting
        cos = cos.reshape(1, 1, seq_len, head_size // 2)
        sin = sin.reshape(1, 1, seq_len, head_size // 2)

        # Разделяем на четные и нечетные компоненты по ПОСЛЕДНЕМУ измерению
        x_even = x[..., 0::2]  # [batch_size, num_heads, seq_len, head_size//2]
        x_odd = x[..., 1::2]   # [batch_size, num_heads, seq_len, head_size//2]

        # Применяем поворот: q' = q * cos(mθ) + rotate(q) * sin(mθ)
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Объединяем обратно в исходную размерность
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # [batch_size, seq_len, head_size]

        return x_rotated


import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


    
class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        window_size: int,
        rope: RoPE = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._num_heads = num_q_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope
        self._window_size = window_size

        self._q = nn.Linear(emb_size, self._num_heads * head_size)
        self._k = nn.Linear(emb_size, num_kv_heads * head_size)
        self._v = nn.Linear(emb_size, num_kv_heads * head_size)

        # Создание causal маски
        mask = self._create_sliding_window_mask(max_seq_len, self._window_size)
        self.register_buffer(
            "_tril_mask", mask.bool() if hasattr(torch, "bool") else mask.byte()
        )
        
        self._layer = nn.Linear(head_size * self._num_heads, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        use_cache: bool = True,
        cache: list = None,
    ):
        batch_size, seq_len, emb_size = x.shape

        if seq_len > self._max_seq_len:
            raise ValueError(
                f"Длина последовательности {seq_len} превышает максимум {self._max_seq_len}"
            )

        # Пропустите тензор x через матрицы Wq, Wk , Wv, чтобы получить матрицы запроса, ключа и значения.
        k = self._k(x)  # [B, T, hs]
        q = self._q(x)  # [B, T, hs]
        v = self._v(x)  # [B, T, hs]

        # Шаг 2: Изменение формы для multi-head
        # [batch_size, seq_len, num_heads * head_size] 
        # -> [batch_size, seq_len, num_heads, head_size]
        # Измените форму запроса (query) на batch_size × num_q_heads × seq_len × head_size.
        q = q.reshape(batch_size, seq_len, self._num_heads, self._head_size)

        # Измените форму ключа (key) и значения (value) на batch_size × num_kv_heads × seq_len × head_size.
        k = k.reshape(batch_size, seq_len, self._num_kv_heads, self._head_size)
        v = v.reshape(batch_size, seq_len, self._num_kv_heads, self._head_size)
     

        # 3. Transpose: [B, T, H, hs] -> [B, H, T, hs]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        start_pos = 0
        if cache is not None:
            k_cache, v_cache = cache
            cache_len = k_cache.shape[2]
            start_pos = cache_len
        
        # Пропустите матрицы запроса и ключа через экземпляр rope, чтобы выполнить поворот.
        if self._rope is not None:
            # Применяем RoPE к Q и K (НЕ к V!)
            q = self._rope(q, start_pos=start_pos)  # [B, T, hs]
            k = self._rope(k, start_pos=start_pos)  # [B, T, hs]

        # Если cache пришел, то объединяем кэш и одну строку из ключа и значения. Это будут новые key и value  для последующих вычислений.
        # 5. Кэширование (для autoregressive generation)
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)  # Concat по seq_len (dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Если use_cache == True, то сохраните матрицы ключа и значения для кэша (это нужно сделать до дублирования голов).
        #if use_cache == True:
        #    # Обрезаем до последних window_size токенов
        #    k_to_cache = k[:, :, -self._window_size:, :]
        #    v_to_cache = v[:, :, -self._window_size:, :]
        #    kv_cache = (k_to_cache, v_to_cache)

        # Продублируйте головы в тензорах ключа (key) и значения (value), чтобы получился тензор размера на batch_size × num_q_heads × seq_len × head_size.
        #k = self._repeat_kv_heads(k, self._num_heads, self._num_kv_heads)
        #v = self._repeat_kv_heads(v, self._num_heads, self._num_kv_heads)
        k_expanded = self._repeat_kv_heads(k, self._num_heads, self._num_kv_heads)
        v_expanded = self._repeat_kv_heads(v, self._num_heads, self._num_kv_heads)
        
        # Перемножим матрицы запроса и ключа (транспонированную), чтобы вычислить матрицу внимания.
        # И разделить все значения в матрице внимания на корень из head_size.
        scores = q @ k_expanded.transpose(-2, -1) / (self._head_size ** 0.5)

        # 8. Применение маски
        k_seq_len = k_expanded.size(2)  # Длина K после concat с кэшем
    
        if cache is None:
            # Случай 1: Без кэша - полная квадратная маска
            # scores: [B, H, seq_len, seq_len]
            # Применяем маску [:seq_len, :seq_len]
            scores = scores.masked_fill(
                ~self._tril_mask[:seq_len, :seq_len], 
                float("-inf")
            )

        # Применить к матрице внимания (построчно) функцию Softmax.
        weights = F.softmax(scores, dim=-1)

        # Перемножим матрицу внимания и матрицу значения.
        x_out = weights @ v_expanded  # [B, T, hs]

        # Измените форму тензора на batch_size × seq_len × num_heads*head_size.
        # Transpose обратно и concatenate heads
        x_out = x_out.transpose(1, 2)  # [B, T_q, H, hs]
        x_out = x_out.contiguous()  # Важно для reshape!
        concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_heads * self._head_size)

        #concatenated_attention = x_out.reshape(batch_size, seq_len, self._num_heads * self._head_size)

        # Пропустите получившийся тензор через последний линейный слой.
        # 3. Проецируем в пространство эмбеддингов
        projected_output = self._layer(concatenated_attention)

        # 4. Применяем dropout для регуляризации
        output = self._dropout(projected_output)

        if use_cache:
            # Обрезаем оригинальный K и V (до дублирования)
            k_to_cache = k[:, :, -self._window_size:, :]
            v_to_cache = v[:, :, -self._window_size:, :]
            kv_cache = (k_to_cache, v_to_cache)
            return output, kv_cache
        else:
            return output, None

    def _repeat_kv_heads(
        self,
        kv: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int
    ) -> torch.Tensor:
        """
        Дублирует головы K/V для соответствия количеству голов Q.

        Args:
            kv: [batch_size, num_kv_heads, seq_len, head_size]
            num_q_heads: Количество голов Query (например, 8)
            num_kv_heads: Количество голов Key/Value (например, 2)

        Returns:
            [batch_size, num_q_heads, seq_len, head_size]

        Example:
            num_q_heads=8, num_kv_heads=2
            Каждая голова KV дублируется 4 раза:
            [KV0, KV1] -> [KV0, KV0, KV0, KV0, KV1, KV1, KV1, KV1]
        """
        batch_size, num_kv_heads, seq_len, head_size = kv.shape

        if num_q_heads == num_kv_heads:
            # Нет необходимости дублировать
            return kv

        # Вычисляем сколько раз нужно повторить каждую голову
        num_repeats = num_q_heads // num_kv_heads

        # repeat_interleave дублирует каждую голову num_repeats раз
        # [B, num_kv_heads, S, hs] -> [B, num_q_heads, S, hs]
        # [B, num_kv_heads, S, hs] -> [B, num_kv_heads, 1, S, hs]
        kv = kv.unsqueeze(2)
        
        # [B, num_kv_heads, 1, S, hs] -> [B, num_kv_heads, num_repeats, S, hs]
        kv = kv.repeat(1, 1, num_repeats, 1, 1)
        
        # [B, num_kv_heads, num_repeats, S, hs] -> [B, num_q_heads, S, hs]
        kv = kv.reshape(batch_size, num_q_heads, seq_len, head_size)
        

        return kv

    def _create_sliding_window_mask(
        self,
        max_seq_len: int,
        window_size: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Создает маску для Sliding Window Attention.

        Args:
            max_seq_len: Максимальная длина последовательности
            window_size: Размер окна внимания
            device: Устройство для размещения тензора

        Returns:
            Маска формы [max_seq_len, max_seq_len], где True = разрешено

        Example:
            >>> mask = create_sliding_window_mask(8, 3)
            >>> print(mask.int())
            tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1]])
        """
        row_indices = torch.arange(max_seq_len, device=device).unsqueeze(1)  # [max_seq_len, 1]
        col_indices = torch.arange(max_seq_len, device=device).unsqueeze(0)  # [1, max_seq_len]

        causal_mask = col_indices <= row_indices

        window_mask = (row_indices - col_indices) <= window_size

        mask = causal_mask & window_mask
        
        return mask

class Decoder(nn.Module):
    def __init__(self, 
        num_q_heads: int,
        num_kv_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        window_size: int,
        rope: RoPE,
        dropout: float = 0.1
    ):
        super().__init__()
        self._heads = GroupedQueryAttention(
            num_q_heads=num_q_heads, 
            num_kv_heads=num_kv_heads,
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len,
            window_size=window_size,
            rope=rope,
            dropout=dropout
        )
        self._ff = SwiGLU(emb_size=emb_size, dropout=dropout)
        self._norm1 = RMSNorm(emb_size)
        self._norm2 = RMSNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None) -> torch.Tensor:
        norm1_out = self._norm1(x)
        attention, kv_caches = self._heads(norm1_out, mask, use_cache=use_cache, cache=cache)
        out = attention + x
        
        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)

        if use_cache is True:
            return (ffn_out + out, kv_caches)
        else:
            return (ffn_out + out, None)



from torch import nn
import torch
import torch.nn.functional as F

class Mistral(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        

        self._max_seq_len = config["max_position_embeddings"]
        # Инициализация слоев
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], 
            emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=config["embed_dim"] // config["num_q_heads"],
            max_seq_len=config["max_position_embeddings"]
        )
        #self._position_embeddings = PositionalEmbeddings(
        #    max_seq_len=max_seq_len, 
        #    emb_size=emb_size
        #)
        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList([Decoder(
            num_q_heads=config["num_q_heads"],
            num_kv_heads=config["num_kv_heads"],
            emb_size=config["embed_dim"],
            head_size=config["embed_dim"] // config["num_q_heads"],
            max_seq_len=config["max_position_embeddings"],
            window_size=config["window_size"],
            rope=self._position_embeddings,
            dropout=config["dropout"] 
        ) for _ in range(config["num_layers"])])
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None) -> tuple:
        # Проверка длины последовательности (только при отсутствии кэша)
        if cache is None and x.size(1) > self._max_seq_len:
            raise ValueError(f"Длина последовательности {x.size(1)} превышает максимальную {self.max_seq_len}")
        
        # Эмбеддинги токенов и позиций
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]
       #pos_out = self._position_embeddings(x)  # [batch, seq_len, emb_size]
        
        # Комбинирование
        out = self._dropout(tok_out)  # [batch, seq_len, emb_size]
        
        # Стек декодеров с передачей кэша
        new_cache = []
        for i, decoder in enumerate(self._decoders):
            decoder_cache = cache[i] if cache is not None else None
            decoder_result = decoder(out, use_cache=use_cache, cache=decoder_cache)

            # Извлекаем результат из кортежа
            if use_cache:
                out, decoder_new_cache = decoder_result
                new_cache.append(decoder_new_cache)
            else:
                out = decoder_result[0]

        out = self._norm(out)
        logits = self._linear(out)
            
        # Возвращаем результат с учетом use_cache
        if use_cache:
            return (logits, new_cache)
        else:
            return (logits, None)

    def generate(self,
        x: torch.Tensor, 
        max_new_tokens: int, 
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        cache = None

        for _ in range(max_new_tokens):
            if use_cache and cache is not None:
                # Используем кэш - передаем только последний токен
                x_input = x[:, -1:]  # [batch_size, 1]
            else:
                # Первая итерация или кэш отключен - передаем всю последовательность
                x_input = x
            
            # Прямой проход с кэшем
            logits, new_cache = self.forward(x_input, use_cache=use_cache, cache=cache)
            
            # Обновляем кэш для следующей итерации
            if use_cache:
                cache = new_cache

            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Масштабируем логиты температурой
            if temperature > 0:
                logits_scaled = last_logits / temperature
            else:
                logits_scaled = last_logits

            if do_sample == True and top_k != None:
                _, topk_indices = torch.topk(logits_scaled, top_k, dim=-1)

                # # Заменим все НЕ top-k логиты на -inf
                masked_logits = logits_scaled.clone()
                vocab_size = logits_scaled.size(-1)

                # создаём маску: 1, если токен НЕ в topk_indices
                mask = torch.ones_like(logits_scaled, dtype=torch.uint8)
                mask.scatter_(1, topk_indices, 0)  # 0 там, где top-k индексы
                masked_logits[mask.byte()] = float('-inf')

                logits_scaled = masked_logits

            if do_sample == True and top_p != None:
                # 1. Применим softmax, чтобы получить вероятности:
                probs = F.softmax(logits_scaled, dim=-1)  # [B, vocab_size]
                # 2. Отсортируем токены по убыванию вероятностей:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # 3. Посчитаем кумулятивную сумму вероятностей:
                cum_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, vocab_size]
                # 4. Определим маску: оставить токены, пока сумма < top_p
                sorted_mask = (cum_probs <= top_p).byte()  # [B, vocab_size]
                # Гарантируем, что хотя бы первый токен останется
                sorted_mask[:, 0] = 1
                # 5. Преобразуем маску обратно в оригинальный порядок:
                # Создаём полную маску из 0
                mask = torch.zeros_like(probs, dtype=torch.uint8)
                # Устанавливаем 1 в местах нужных токенов
                mask.scatter_(dim=1, index=sorted_indices, src=sorted_mask)
                # 6. Зануляем логиты токенов вне топ-p:
                logits_scaled[~mask] = float('-inf')

            # 4. Применяем Softmax
            probs = F.softmax(logits_scaled, dim=-1)  # [batch_size, vocab_size]


            if do_sample == True:
                # 5. Если do_sample равен True, то отбираем токен случайно с помощью torch.multinomial
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            else:
                # 5. Если do_sample равен False, то выбираем токен с максимальной вероятностью
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 6. Добавляем его к последовательности
            x = torch.cat([x, next_token], dim=1)  # [batch_size, seq_len+1]
        return x

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self._vocab_size,
            'max_seq_len': self._max_seq_len,
            'emb_size': self._emb_size,
            'num_heads': self._num_heads,
            'head_size': self._head_size,
            'num_layers': self._num_layers
        }, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len