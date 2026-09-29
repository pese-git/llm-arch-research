"""
Базовый абстрактный класс для всех больших языковых моделей (LLM).

Научная суть:
Модели типа LLM строятся по модульному принципу — конкретные GPT, LLaMA и др. должны наследоваться от этого класса и реализовывать базовый набор интерфейсов для совместимости с training loop, генерацией, инференсом и т.д.

Пользовательский уровень:
Базовый интерфейс минимизирует дублирование кода и позволяет быстро добавлять новые архитектуры.

Использование:
    class MyModel(BaseModel):
        ...
    model = MyModel(config)
    logits, cache = model(input_ids)
    tokens = model.generate(input_ids, max_new_tokens=20, do_sample=False)

Наследник реализует forward (и задаёт self._max_seq_len); generate, save и load общие для всех моделей.
"""
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch

from llm.core.generation import (
    check_attention_mask,
    next_generation_input,
    sample_next_token,
    validate_sampling_args,
)


class BaseModel(nn.Module, ABC):
    """
    Абстрактный класс — стандарт для всех архитектур LLM.

    Научная идея:
    Реализация унифицированного входа/выхода для поддержки построения и обучения любых современных языковых моделей.

    Args:
        config (dict): Параметры архитектуры (размерность эмбеддингов, число слоев, heads и т.д.)

    Attributes:
        config (dict): Конфиг модели
    """

    def __init__(self, config: dict):
        """
        Инициализация модели.

        Args:
            config (dict): Настройки архитектуры модели (размеры слоев, типы блоков и т.д.)
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Прямой проход — получение логитов для входных токенов.

        Args:
            x (Tensor[int]): Индексы токенов [batch, seq_len]
            use_cache (bool): Вернуть KV-кэш для продолжения генерации
            cache (Optional[list]): KV-кэш предыдущих токенов (по слою на элемент)
            attention_mask (Optional[Tensor]): Маска паддинга [batch, seq_len]; поддерживается
                только правый паддинг (см. docs/masks.md)
        Returns:
            (logits, cache): логиты [batch, seq_len, vocab_size] и новый кэш
            (None при use_cache=False)
        """
        pass

    def save(self, path: str) -> None:
        """
        Сохраняет модель в один файл: класс, конфиг и веса.

        Вычисляемые буферы (маски attention, таблицы RoPE) не сохраняются — они
        строятся заново из конфига при загрузке.

        Args:
            path: Путь к файлу (например, "checkpoints/model.pt").

        Пример:
            >>> model.save("checkpoints/mistral.pt")
            >>> restored = Mistral.load("checkpoints/mistral.pt")
        """
        torch.save(
            {
                "model_class": type(self).__name__,
                "config": dict(self.config),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseModel":
        """
        Загружает модель, сохранённую методом save.

        Модель создаётся заново по сохранённому конфигу, поэтому аргументы конструктора
        передавать не нужно. Файл читается с weights_only=True: в нём только тензоры и
        простые значения конфига, произвольный код при загрузке не выполняется.

        Args:
            path: Путь к файлу, созданному save.
            device: Устройство для модели ("cpu", "cuda", ...).

        Returns:
            Модель класса cls в режиме eval (как после обучения перед инференсом).

        Raises:
            ValueError: Если файл не создан save (например, это голый state_dict) или
                сохранён моделью другого класса.
        """
        # Читаем на CPU и переносим уже собранную модель: так файл, сохранённый на GPU,
        # грузится и без GPU, а модель целиком оказывается на device
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {"config", "state_dict"} <= checkpoint.keys():
            raise ValueError(
                f"{path} не похож на файл BaseModel.save: нет ключей 'config' и 'state_dict'. "
                "Голый state_dict загружается через cls(config).load_state_dict(torch.load(path))."
            )
        saved_class = checkpoint.get("model_class")
        if saved_class is not None and saved_class != cls.__name__:
            raise ValueError(
                f"{path} сохранён моделью {saved_class}, а загружается как {cls.__name__}"
            )
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()

    def auxiliary_loss(self) -> Optional[torch.Tensor]:
        """
        Вспомогательный loss последнего прямого прохода, который нужно прибавить к loss
        языковой модели при обучении, или None, если его нет.

        По умолчанию None. Mixtral возвращает load-balancing loss роутера MoE, если в
        конфиге задан router_aux_loss_coef > 0.
        """
        return None

    @property
    def max_seq_len(self) -> int:
        """Максимальная длина последовательности (max_position_embeddings)."""
        return self._max_seq_len

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Авторегрессивная генерация — общая для всех архитектур.

        На каждом шаге модель считает логиты последней позиции, из них выбирается
        следующий токен (sample_next_token), и он дописывается к последовательности.
        С кэшем в forward подаётся только новый токен. Когда последовательность
        становится длиннее max_seq_len, модель продолжает по последним max_seq_len
        токенам без кэша (next_generation_input). Градиенты не считаются.

        Args:
            x: Промпт [batch, seq_len].
            max_new_tokens: Сколько токенов сгенерировать (не больше — меньше при eos_token_id).
            do_sample: False — жадный выбор самого вероятного токена;
                True — сэмплирование с temperature, top_k или top_p.
            temperature: Температура (> 0); меньше 1 — распределение острее, больше 1 — ровнее.
            top_k: Сэмплировать только из top_k самых вероятных токенов.
            top_p: Nucleus sampling — из минимального набора токенов с суммарной вероятностью ≥ top_p.
            use_cache: Использовать KV-кэш (результат тот же, генерация быстрее).
            attention_mask: Маска промпта [batch, seq_len]; допускается только из единиц
                (генерация после паддинга не поддерживается).
            eos_token_id: Токен конца текста. Строка, сгенерировавшая его, считается
                законченной; генерация останавливается, когда закончены все строки.
            pad_token_id: Чем заполнять законченные строки, пока генерируют остальные
                (по умолчанию eos_token_id).

        Returns:
            Промпт с дописанными токенами [batch, seq_len + число_шагов].

        Raises:
            ValueError: Если при do_sample=True temperature ≤ 0, заданы одновременно
                top_k и top_p, top_k ≤ 0 или top_p вне (0, 1].
            NotImplementedError: Если в attention_mask есть нули.
            TypeError: Если передан неизвестный именованный аргумент.

        Примеры:
            >>> model.generate(x, max_new_tokens=20, do_sample=False)                   # greedy
            >>> model.generate(x, max_new_tokens=20, do_sample=True, temperature=0.8)
            >>> model.generate(x, max_new_tokens=20, do_sample=True, top_k=50)
            >>> model.generate(x, max_new_tokens=20, do_sample=True, top_p=0.9, eos_token_id=3)

        Ссылки:
            - Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus sampling):
              https://arxiv.org/abs/1904.09751
        """
        validate_sampling_args(do_sample, temperature, top_k, top_p)
        check_attention_mask(attention_mask, x, generating=True)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        finished = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        cache = None
        for _ in range(max_new_tokens):
            # С кэшем подаём только последний токен; за пределами max_seq_len берём
            # последние max_seq_len токенов и пересчитываем без кэша.
            x_input, cache = next_generation_input(x, cache, use_cache, self.max_seq_len)
            logits, new_cache = self(x_input, use_cache=use_cache, cache=cache)
            if use_cache:
                cache = new_cache

            next_token = sample_next_token(
                logits[:, -1, :], do_sample, temperature, top_k, top_p
            )  # [batch, 1]

            if eos_token_id is not None:
                next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_token_id)
                finished |= next_token.squeeze(-1) == eos_token_id

            x = torch.cat([x, next_token], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return x
