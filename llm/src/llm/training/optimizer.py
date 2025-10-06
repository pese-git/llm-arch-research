import torch.optim as optim


def get_optimizer(model, lr=3e-4, weight_decay=0.01, optimizer_type="adamw"):
    """
    Возвращает оптимизатор для обучения модели.
    """
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")
