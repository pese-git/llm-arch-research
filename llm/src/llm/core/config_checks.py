"""
Проверки конфигурации моделей: размер головы attention и число голов.

Неверный конфиг должен давать понятную ошибку в конструкторе, а не молча урезанное
пространство внимания или непонятный RuntimeError в первом forward.
"""


def resolve_head_size(config: dict, num_heads_key: str, rope: bool = False) -> int:
    """
    Размер одной головы attention из конфига.

    Если в конфиге задан head_size, используется он; тогда num_heads * head_size может
    отличаться от embed_dim — выходная проекция attention возвращает результат в embed_dim.
    Иначе head_size = embed_dim // num_heads, и embed_dim обязан делиться на число голов:
    при остатке внимание молча работало бы в пространстве меньше embed_dim.

    Args:
        config: Конфиг модели (embed_dim, число голов, опционально head_size).
        num_heads_key: Ключ числа голов (Q-голов) в конфиге: "num_heads" или "num_q_heads".
        rope: Модель использует RoPE — тогда head_size должен быть чётным
            (RoPE поворачивает пары координат).

    Raises:
        ValueError: Если число голов или head_size не положительные, embed_dim не делится
            на число голов (без явного head_size) или head_size нечётный при RoPE.
    """
    embed_dim = config["embed_dim"]
    num_heads = config[num_heads_key]
    if num_heads < 1:
        raise ValueError(f"{num_heads_key} должно быть ≥ 1, получено {num_heads}")

    head_size = config.get("head_size")
    if head_size is None:
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} не делится на {num_heads_key}={num_heads}: "
                f"размер головы был бы {embed_dim // num_heads}, и внимание работало бы в "
                f"{embed_dim // num_heads * num_heads} измерениях вместо {embed_dim}. "
                "Подберите делящиеся значения или задайте head_size явно."
            )
        head_size = embed_dim // num_heads
    elif head_size < 1:
        raise ValueError(f"head_size должен быть ≥ 1, получено {head_size}")

    if rope and head_size % 2 != 0:
        raise ValueError(
            f"head_size={head_size} (embed_dim={embed_dim}, {num_heads_key}={num_heads}) "
            "должен быть чётным: RoPE поворачивает пары координат"
        )
    return head_size
