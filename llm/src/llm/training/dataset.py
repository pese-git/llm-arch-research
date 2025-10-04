import torch
from torch.utils.data import Dataset
from typing import List, Any


class TextDataset(Dataset):
    """
    Простой датасет для языкового моделирования (LLM).
    Работает с любым токенизатором, реализующим интерфейс BaseTokenizer.
    """

    def __init__(self, texts: List[str], tokenizer: Any, block_size: int = 128):
        """
        Инициализация датасета.
        
        Args:
            texts: Список текстов для обучения
            tokenizer: Токенизатор с методами encode/decode
            block_size: Максимальная длина последовательности
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        for text in texts:
            # Кодируем текст в токены
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            
            # Обрезаем или дополняем до нужной длины
            if len(input_ids) > block_size:
                input_ids = input_ids[:block_size]
            else:
                # Дополняем pad_token_id
                pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
                input_ids = input_ids + [pad_token_id] * (block_size - len(input_ids))
            
            self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


class StreamingTextDataset(Dataset):
    """
    Датасет для потоковой обработки больших текстов.
    Токенизация происходит на лету, что экономит память.
    """
    
    def __init__(self, texts: List[str], tokenizer: Any, block_size: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Получаем pad_token_id из токенизатора
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 0)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Токенизация на лету
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Обрезаем или дополняем до нужной длины
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        else:
            input_ids = input_ids + [self.pad_token_id] * (self.block_size - len(input_ids))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        return {"input_ids": input_ids, "labels": labels}


class TextDatasetWithSpecialTokens(TextDataset):
    """
    Расширенная версия TextDataset с поддержкой специальных токенов.
    """
    
    def __init__(self, texts: List[str], tokenizer: Any, block_size: int = 128, 
                 add_bos: bool = False, add_eos: bool = False):
        """
        Args:
            texts: Список текстов
            tokenizer: Токенизатор
            block_size: Максимальная длина
            add_bos: Добавлять токен начала последовательности
            add_eos: Добавлять токен конца последовательности
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.add_bos = add_bos
        self.add_eos = add_eos

        for text in texts:
            # Кодируем с специальными токенами
            input_ids = tokenizer.encode(
                text, 
                add_special_tokens=True,
                add_bos_token=add_bos,
                add_eos_token=eos
            )
            
            # Учитываем специальные токены при обрезке/дополнении
            effective_block_size = block_size
            if add_bos:
                effective_block_size -= 1
            if add_eos:
                effective_block_size -= 1
            
            if len(input_ids) > effective_block_size:
                input_ids = input_ids[:effective_block_size]
            
            # Добавляем специальные токены если нужно
            if add_bos and hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                input_ids = [tokenizer.bos_token_id] + input_ids
            if add_eos and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                input_ids = input_ids + [tokenizer.eos_token_id]
            
            # Дополняем до полной длины
            pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
            if len(input_ids) < block_size:
                input_ids = input_ids + [pad_token_id] * (block_size - len(input_ids))
            
            self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}
