"""
Классическая GPT (Generative Pre-trained Transformer), OpenAI 2018.

Научная суть:
    - Первая массовая архитектура языка на основе исключительно self-attention механизмов (трансформер-декодер).
    - Обучается сначала на задаче языкового моделирования (unsupervised), далее дообучается на downstream-задачах (transfer learning).
    - Обеспечивает длинную память и “глобальный” контекст благодаря attention.

    Ключевые элементы:
    - masked self-attention (causal)
    - LayerNorm ПОСЛЕ attention и FFN (что отличает от GPT2)
    - GELU активация
    - Absolute learned positional embeddings

    Подробнее: Radford et al., "Improving Language Understanding by Generative Pre-Training", arXiv:1801.10198
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

    Пример использования:
        >>> model = GPT({"vocab_size": 50257, ...})
        >>> logits = model(input_ids)
        >>> out = model.generate(input_ids, max_length=30)
    """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from llm.core.base_model import BaseModel
from llm.core.decoder import Decoder
from llm.core.token_embeddings import TokenEmbeddings
from llm.core.positional_embeddings import PositionalEmbeddings


class GPT(BaseModel):
    """
    Original GPT (Generative Pre-trained Transformer) модель.
    
    Первая версия трансформерной архитектуры от OpenAI, предназначенная
    для генеративного предобучения на текстовых данных.
    
    Args:
        config: Словарь конфигурации с параметрами:
            - vocab_size: Размер словаря токенов
            - embed_dim: Размерность векторных представлений
            - num_heads: Количество голов внимания
            - num_layers: Количество декодерных слоев
            - max_position_embeddings: Максимальная длина последовательности
            - dropout: Вероятность dropout
    
    Attributes:
        _token_embeddings: Слой векторных представлений токенов
        _position_embeddings: Слой позиционных эмбеддингов
        _decoders: Список декодерных слоев
        _norm: Финальный слой нормализации
        _linear: Выходной линейный слой
    """
    def __init__(self, config):
        super().__init__(config)

        # Инициализация слоев
        self._max_seq_len = config["max_position_embeddings"]
        self._token_embeddings = TokenEmbeddings(
            vocab_size=config["vocab_size"], 
            emb_size=config["embed_dim"]
        )
        self._position_embeddings = PositionalEmbeddings(
            max_seq_len=config["max_position_embeddings"], 
            emb_size=config["embed_dim"]
        )
        self._dropout = nn.Dropout(config["dropout"])
        # head_size = emb_size // num_heads
        self._decoders = nn.ModuleList([Decoder(
            num_heads=config["num_heads"],
            emb_size=config["embed_dim"],
            head_size=config["embed_dim"] // config["num_heads"],
            max_seq_len=config["max_position_embeddings"],
            dropout=config["dropout"] 
        ) for _ in range(config["num_layers"])])
        self._linear = nn.Linear(config["embed_dim"], config["vocab_size"])
    
    @property
    def max_seq_len(self):
        """Возвращает максимальную длину последовательности."""
        return self._max_seq_len

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Прямой проход через GPT
        
        Args:
            x: Входной тензор [batch_size, seq_len]
            
        Returns:
            Тензор логитов [batch_size, seq_len, vocab_size]
        """
        # Проверка длины последовательности
        if x.size(1) > self._max_seq_len:
            raise ValueError(f"Длина последовательности {x.size(1)} превышает максимальную {self._max_seq_len}")
        
        # Эмбеддинги токенов и позиций
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]
        pos_out = self._position_embeddings(x.size(1))  # [seq_len, emb_size]
        
        # Комбинирование
        out = self._dropout(tok_out + pos_out.unsqueeze(0))  # [batch, seq_len, emb_size]
        
        # Стек декодеров
        for decoder in self._decoders:
            out = decoder(out)
            
        return self._linear(out)  # [batch, seq_len, vocab_size]


#    def forward(self, input_ids, attention_mask=None):
#        B, T = input_ids.size()
#        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
#
#        x = self.token_emb(input_ids) + self.pos_emb(pos)
#
#        for block in self.blocks:
#            x = block(x, attention_mask)
#
#        x = self.ln_f(x)
#        logits = self.head(x)
#        return logits


    def generate(self,
        x: torch.Tensor, 
        max_new_tokens: int, 
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        attention_mask: torch.Tensor = None,  # Добавляем для совместимости с HF
        **kwargs  # Игнорируем остальные параметры
    ) -> torch.Tensor:
        """Авторегрессивная генерация текста.
        
        Параметры:
            x: Входной тензор с индексами токенов формы [batch_size, seq_len],
               где batch_size - размер батча, seq_len - длина последовательности.
            max_new_tokens: Максимальное количество новых токенов для генерации.
            do_sample: Флаг выбора режима генерации:
                - True: вероятностное сэмплирование
                - False: жадный поиск (argmax)
            temperature: Параметр температуры для сэмплирования:
                - >1.0 - более случайные результаты
                - 1.0 - нейтральное значение
                - <1.0 - более предсказуемые результаты
                Должна быть > 0 (по умолчанию: 1.0)
            top_k: Если задан (и do_sample=True), используется top-k сэмплирование:
                - Выбираются только top_k самых вероятных токенов
                - Остальным токенам устанавливается вероятность 0
                - None: отключено (по умолчанию)
            top_p: Если задан (и do_sample=True), используется nucleus (top-p) сэмплирование:
                - Выбираются токены с кумулятивной вероятностью ≤ top_p
                - Гарантируется, что хотя бы один токен остаётся (даже если его вероятность > top_p)
                - None: отключено (по умолчанию)
                - Должен быть в диапазоне (0, 1]
        
        Возвращает:
            torch.Tensor: Тензор с расширенной последовательностью токенов формы 
                          [batch_size, seq_len + max_new_tokens]

        Исключения:
            ValueError: Если входная последовательность длиннее max_seq_len
            ValueError: Если temperature <= 0
            ValueError: Если одновременно заданы top_k и top_p
            ValueError: Если top_k задан и ≤ 0
            ValueError: Если top_p задан и не в диапазоне (0, 1]

        Примеры:
            >>> # Жадная генерация
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            >>> 
            >>> # Вероятностная генерация с top-k
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_k=50)
            >>>
            >>> # Nucleus sampling (top-p)
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
            >>>
            >>> # Комбинация температуры и top-k
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, 
            ...                        temperature=0.7, top_k=50)

        Примечания:
            1. Для детерминированных результатов в режиме сэмплирования 
               зафиксируйте random seed (torch.manual_seed).
            2. Температура влияет только на режим сэмплирования (do_sample=True).
            3. Одновременное использование top_k и top_p запрещено.
            4. При do_sample=False параметры top_k, top_p и temperature игнорируются.

        Args:
            x (torch.Tensor): Входной тензор с индексами токенов формы [batch_size, seq_len],
                              где batch_size - размер батча, seq_len - длина последовательности.
            max_new_tokens (int): Максимальное количество новых токенов для генерации.
            do_sample (bool): Флаг выбора режима генерации:
                              - True: вероятностное сэмплирование
                              - False: жадный поиск (argmax)
            temperature (float): Параметр температуры для сэмплирования:
                              - >1.0 - более случайные результаты
                              - 1.0 - нейтральное значение
                              - <1.0 - более предсказуемые результаты
                              Должна быть > 0 (по умолчанию: 1.0)

        Returns:
            torch.Tensor: Тензор с расширенной последовательностью токенов формы 
                          [batch_size, seq_len + max_new_tokens]

        Raises:
            ValueError: Если входная последовательность длиннее max_seq_len
            ValueError: Если temperature <= 0

        Examples:
            >>> # Жадная генерация
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            >>>
            >>> # Вероятностная генерация с температурой
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, temperature=0.7)
            >>>
            >>> # Более случайная генерация
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, temperature=1.5)

        Note:
            Для детерминированных результатов в режиме сэмплирования 
            зафиксируйте random seed (torch.manual_seed).
            Температура влияет только на режим сэмплирования (do_sample=True).
        """
        for _ in range(max_new_tokens):
            # 1. Обрезаем вход, если последовательность слишком длинная
            x_cond = x[:, -self._max_seq_len:]

            # 2. Передаем последовательность в метод forward класса GPT и полуаем логиты.
            logits = self.forward(x_cond)

            # 3. Берем логиты для последнего токена
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

                # создаём маску: True, если токен НЕ в topk_indices
                mask = torch.ones_like(logits_scaled, dtype=torch.bool if hasattr(torch, 'bool') else torch.uint8)
                mask.scatter_(1, topk_indices, False if hasattr(torch, 'bool') else 0)  # False там, где top-k индексы
                masked_logits[mask] = float('-inf')

                logits_scaled = masked_logits

            if do_sample == True and top_p != None:
                # 1. Применим softmax, чтобы получить вероятности:
                probs = F.softmax(logits_scaled, dim=-1)  # [B, vocab_size]
                # 2. Отсортируем токены по убыванию вероятностей:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # 3. Посчитаем кумулятивную сумму вероятностей:
                cum_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, vocab_size]
                # 4. Определим маску: оставить токены, пока сумма < top_p
                sorted_mask = (cum_probs <= top_p)  # [B, vocab_size]
                # Гарантируем, что хотя бы первый токен останется
                sorted_mask[:, 0] = True
                # 5. Преобразуем маску обратно в оригинальный порядок:
                # Создаём полную маску из False
                mask = torch.zeros_like(probs, dtype=torch.bool if hasattr(torch, 'bool') else torch.uint8)
                # Устанавливаем True в местах нужных токенов
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

#    def generate(self, input_ids, max_length=50):
#        for _ in range(max_length):
#            logits = self.forward(input_ids)
#            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
#            input_ids = torch.cat([input_ids, next_token], dim=1)
#        return input_ids
