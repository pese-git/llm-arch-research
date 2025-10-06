import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from math import sqrt
from torch import nn
import torch
import math

from llm.core.base_model import BaseModel
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.rope import RoPE
from llm.core.swi_glu import SwiGLU
from llm.core.rms_norm import RMSNorm
from llm.core.gelu import GELU

    
class HeadAttention(nn.Module):

    def __init__(self, emb_size: int, head_size: int, max_seq_len: int, rope: RoPE):
        super().__init__()
        self._emb_size = emb_size
        self._head_size = head_size
        self._max_seq_len = max_seq_len
        self._rope = rope

        self._k = nn.Linear(emb_size, head_size)
        self._q = nn.Linear(emb_size, head_size)
        self._v = nn.Linear(emb_size, head_size)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('_tril_mask', mask.bool() if hasattr(torch, 'bool') else mask.byte())

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None) -> tuple:
        seq_len = x.shape[1]
        if seq_len > self._max_seq_len:
            raise ValueError(f"Длина последовательности {seq_len} превышает максимум {self._max_seq_len}")

        k = self._k(x)  # [B, T, hs]
        q = self._q(x)  # [B, T, hs]
        v = self._v(x)  # [B, T, hs]

        # ✅ Применяем RoPE к Q и K (НЕ к V!)
        q = self._rope(q)  # [B, T, hs]
        k = self._rope(k)  # [B, T, hs]

        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=1)  # [B, cache_len + T, hs]
            v = torch.cat([v_cache, v], dim=1)  # [B, cache_len + T, hs]
        
        scores = q @ k.transpose(-2, -1) / sqrt(self._head_size)
        
        if cache is None:
            scores = scores.masked_fill(~self._tril_mask[:seq_len, :seq_len], float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        x_out = weights @ v  # [B, T, hs]

        if use_cache is True:
            return (x_out, (k, v))
        else:
            return (x_out, None)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, rope: RoPE, dropout: float = 0.1):

        super().__init__()
        self._heads = nn.ModuleList([
            HeadAttention(
                emb_size=emb_size, 
                head_size=head_size, 
                max_seq_len=max_seq_len,
                rope=rope,
            ) for _ in range(num_heads)
        ])
        self._layer = nn.Linear(head_size * num_heads, emb_size)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, use_cache: bool = True, cache: list = None):

        attention_results = []
        for i, head in enumerate(self._heads):
            head_cache = cache[i] if cache is not None else None
            result = head(x, use_cache=use_cache, cache=head_cache)
            attention_results.append(result)
        
        outputs, caches = zip(*attention_results)
        attention_outputs = list(outputs)
        kv_caches = list(caches)
 
        concatenated_attention = torch.cat(attention_outputs, dim=-1)

        projected_output = self._layer(concatenated_attention)
        
        final_output = self._dropout(projected_output)
        
        if use_cache is True:
            return (final_output, kv_caches)
        else:
            return (final_output, None)

    
class Decoder(nn.Module):
    def __init__(self, 
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        rope: RoPE,
        dropout: float = 0.1
    ):
        super().__init__()
        self._heads = MultiHeadAttention(
            num_heads=num_heads, 
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len,
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



class Llama(BaseModel):
    def __init__(self,config):
        super().__init__(config)

        # Инициализация слоев
        self._max_seq_len = config["max_position_embeddings"]
        self._token_embeddings = TokenEmbeddings(
         vocab_size=config["vocab_size"], 
            emb_size=config["embed_dim"]
        )
        self._position_embeddings = RoPE(
            head_size=config["embed_dim"] // config["num_heads"],
            max_seq_len=config["max_position_embeddings"]
        )

        self._dropout = nn.Dropout(config["dropout"])
        self._decoders = nn.ModuleList([Decoder(
            num_heads=config["num_heads"],
            emb_size=config["embed_dim"],
            head_size=config["embed_dim"] // config["num_heads"],
            max_seq_len=config["max_position_embeddings"],
            rope=self._position_embeddings,
            dropout=config["dropout"], 
        ) for _ in range(config["num_layers"])])
        self._norm = RMSNorm(config["embed_dim"])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None) -> tuple:
        # Проверка длины последовательности (только при отсутствии кэша)
        if cache is None and x.size(1) > self._max_seq_len:
            raise ValueError(f"Длина последовательности {x.size(1)} превышает максимальную {self.max_seq_len}")
        
        
        # Вычисление start_pos из кэша (если кэш передан)
        #if cache is not None:
        #    # При кэше обрабатываем только один токен (последний)
        #    seq_len = 1
        #    # Вычисляем start_pos из самого нижнего уровня кэша
        #    if cache and cache[0] and cache[0][0]:
        #        key_cache, _ = cache[0][0]  # Первый декодер, первая голова
        #        start_pos = key_cache.size(1)  # cache_len
        #    else:
        #        start_pos = 0
        #else:
        #    # Без кэша работаем как раньше
        #    start_pos = 0
        #    seq_len = x.size(1)

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



    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len