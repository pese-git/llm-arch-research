import torch
from torch import nn, Tensor

class PositionalEmbeddings(nn.Module):
    """
    Класс для создания позиционных эмбеддингов через nn.Embedding.
    
    Позиционные эмбеддинги используются в нейросетях для передачи информации 
    о позиции элементов в последовательности (например, в Transformer).
    
    Особенности:
    - Создаёт обучаемые позиционные эмбеддинги фиксированной длины
    - Поддерживает обработку последовательностей переменной длины
    - Автоматически размещает вычисления на том же устройстве, что и параметры
    
    Args:
        max_seq_len (int): Максимальная длина последовательности
        emb_size (int): Размерность векторного представления позиций
    
    Пример использования:
        >>> pos_encoder = PositionalEmbeddings(max_seq_len=100, emb_size=256)
        >>> # Получить эмбеддинги для последовательности из 10 элементов
        >>> embeddings = pos_encoder(10)  # Tensor shape: [10, 256]
        >>> # Использование в модели
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.pos_emb = PositionalEmbeddings(100, 256)
        ...     def forward(self, x):
        ...         pos = self.pos_emb(x.size(1))
        ...         return x + pos  # Добавляем позиционную информацию
    """

    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=emb_size
        )

    def forward(self, seq_len: int) -> Tensor:
        """
        Возвращает позиционные эмбеддинги для заданной длины последовательности.
        
        Args:
            seq_len (int): Длина последовательности (1 <= seq_len <= max_seq_len)
            
        Returns:
            Tensor: Тензор позиционных эмбеддингов формы [seq_len, emb_size]
            
        Raises:
            IndexError: Если seq_len выходит за допустимые границы
            
        Пример:
            >>> pos_encoder = PositionalEmbeddings(100, 64)
            >>> emb = pos_encoder(10)  # Тензор 10x64
        """
        if seq_len < 1 or seq_len > self.max_seq_len:
            raise IndexError(f"Длина {seq_len} должна быть от 1 до {self.max_seq_len}")
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)

if __name__ == "__main__":
    # Демонстрация работы
    print("Пример использования PositionalEmbeddings:")
    pos_emb = PositionalEmbeddings(max_seq_len=50, emb_size=128)
    
    # Пример 1: Базовое использование
    print("\n1. Базовый пример:")
    emb = pos_emb(10)
    print(f"Форма выходного тензора: {emb.shape}")
    print(f"Среднее значение: {emb.mean().item():.4f}")
    
    # Пример 2: Интеграция с моделью
    print("\n2. Пример интеграции с моделью:")
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_emb = PositionalEmbeddings(50, 128)
            
        def forward(self, x):
            pos = self.pos_emb(x.size(1))
            return x + pos  # Добавляем позиционную информацию
            
    model = DemoModel()
    input_tensor = torch.randn(2, 10, 128)  # [batch, seq, features]
    output = model(input_tensor)
    print(f"Вход: {input_tensor.shape}, Выход: {output.shape}")