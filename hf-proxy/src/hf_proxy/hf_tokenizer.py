"""
Адаптер для интеграции кастомных токенизаторов llm с HuggingFace.
"""

import json
from typing import Dict, List, Optional, Union
from llm.tokenizers import BPETokenizer, BaseTokenizer


class HFTokenizerAdapter:
    """
    Упрощенный адаптер для кастомных токенизаторов llm.
    Предоставляет совместимый с HuggingFace интерфейс.
    """
    
    def __init__(self, llm_tokenizer: BaseTokenizer):
        """
        Инициализация адаптера.
        
        Args:
            llm_tokenizer: Кастомный токенизатор из llm
        """
        self.llm_tokenizer = llm_tokenizer
        
        # Получаем словарь и размер
        self._vocab = llm_tokenizer.get_vocab()
        self.vocab_size = llm_tokenizer.get_vocab_size()
        
        # Устанавливаем специальные токены
        self.pad_token = getattr(llm_tokenizer, 'pad_token', '<pad>')
        self.unk_token = getattr(llm_tokenizer, 'unk_token', '<unk>') 
        self.bos_token = getattr(llm_tokenizer, 'bos_token', '<bos>')
        self.eos_token = getattr(llm_tokenizer, 'eos_token', '<eos>')
        
        # Сохраняем ID специальных токенов
        self.pad_token_id = getattr(llm_tokenizer, 'pad_token_id', 0)
        self.unk_token_id = getattr(llm_tokenizer, 'unk_token_id', 1)
        self.bos_token_id = getattr(llm_tokenizer, 'bos_token_id', 2)
        self.eos_token_id = getattr(llm_tokenizer, 'eos_token_id', 3)
    
    def __call__(self, text: str, **kwargs):
        """
        Вызов токенизатора с параметрами как у HuggingFace.
        
        Args:
            text: Входной текст
            **kwargs: Параметры токенизации
            
        Returns:
            dict: Словарь с токенами
        """
        return_tensors = kwargs.get('return_tensors', None)
        padding = kwargs.get('padding', False)
        truncation = kwargs.get('truncation', False)
        max_length = kwargs.get('max_length', None)
        add_special_tokens = kwargs.get('add_special_tokens', True)
        
        # Кодируем текст
        #input_ids = self.llm_tokenizer.encode(
        #    text, 
        #    add_special_tokens=add_special_tokens
        #)
        if isinstance(text, str):
            input_ids = self.llm_tokenizer.encode(
                text, 
                add_special_tokens=add_special_tokens
            )
            input_ids = [input_ids]  # <-- оборачиваем в batch
        else:
            # Список строк, батч-режим!
            input_ids = [
                self.llm_tokenizer.encode(
                    t,
                    add_special_tokens=add_special_tokens
                ) for t in text
            ]
        
        # Применяем truncation
        if truncation and max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Применяем padding
        if padding and max_length is not None and len(input_ids) < max_length:
            input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))
        
        # Конвертируем в тензоры если нужно
        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor([input_ids])
        
        return {"input_ids": input_ids}
    
    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[List[int], List[List[int]]]:
        """
        Кодирует текст в последовательность токенов.
        
        Args:
            text: Входной текст
            text_pair: Второй текст (для парных задач)
            add_special_tokens: Добавлять специальные токены
            padding: Добавлять паддинг
            truncation: Обрезать последовательность
            max_length: Максимальная длина
            return_tensors: Возвращать тензоры
            
        Returns:
            Список токенов или список списков токенов
        """
        # Кодируем основной текст
        token_ids = self.llm_tokenizer.encode(
            text, 
            add_special_tokens=add_special_tokens
        )
        
        # Обрабатываем text_pair если есть
        if text_pair is not None:
            pair_ids = self.llm_tokenizer.encode(
                text_pair,
                add_special_tokens=False
            )
            token_ids.extend(pair_ids)
        
        # Применяем truncation
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Применяем padding
        if padding and max_length is not None and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        # Конвертируем в тензоры если нужно
        if return_tensors == "pt":
            import torch
            return torch.tensor([token_ids])
        elif return_tensors == "np":
            import numpy as np
            return np.array([token_ids])
        
        return token_ids
    
    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Декодирует последовательность токенов в текст.
        
        Args:
            token_ids: ID токенов
            skip_special_tokens: Пропускать специальные токены
            
        Returns:
            str: Декодированный текст
        """
        # Обрабатываем разные форматы входных данных
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            # Список списков - берем первый элемент
            token_ids = token_ids[0]
        
        # Фильтруем специальные токены если нужно
        if skip_special_tokens:
            special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        return self.llm_tokenizer.decode(token_ids)
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Токенизирует текст в список строковых токенов.
        
        Args:
            text: Входной текст
            
        Returns:
            List[str]: Список токенов
        """
        return self.llm_tokenizer.tokenize(text)
    
    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_attention_mask=None,
        return_tensors=None,
        verbose=True,
    ):
        """
        Pad a list of encoded inputs.
        
        Args:
            encoded_inputs: List of encoded inputs
            padding: Padding strategy
            max_length: Maximum length
            pad_to_multiple_of: Pad to multiple of
            return_attention_mask: Return attention mask
            return_tensors: Return tensors
            verbose: Verbose mode
            
        Returns:
            Padded inputs
        """
        # Простая реализация padding для совместимости
        if isinstance(encoded_inputs, (list, tuple)) and len(encoded_inputs) > 0:
            # Находим максимальную длину
            max_len = 0
            for item in encoded_inputs:
                input_ids = item["input_ids"]
                # Обрабатываем разные типы данных
                if isinstance(input_ids, int):
                    seq_len = 1
                elif hasattr(input_ids, 'shape'):
                    seq_len = input_ids.shape[-1] if len(input_ids.shape) > 1 else len(input_ids)
                else:
                    seq_len = len(input_ids)
                max_len = max(max_len, seq_len)
            
            if max_length is not None:
                max_len = min(max_len, max_length)
            
            # Применяем padding
            for item in encoded_inputs:
                input_ids = item["input_ids"]
                
                # Получаем текущую длину
                if isinstance(input_ids, int):
                    current_len = 1
                elif hasattr(input_ids, 'shape'):
                    current_len = input_ids.shape[-1] if len(input_ids.shape) > 1 else len(input_ids)
                else:
                    current_len = len(input_ids)
                
                if current_len < max_len:
                    # Дополняем pad_token_id
                    padding_length = max_len - current_len
                    
                    # Обрабатываем разные типы данных
                    if isinstance(input_ids, int):
                        item["input_ids"] = [input_ids] + [self.pad_token_id] * padding_length
                    elif hasattr(input_ids, 'shape'):
                        import torch
                        padding_tensor = torch.full((padding_length,), self.pad_token_id, dtype=input_ids.dtype)
                        item["input_ids"] = torch.cat([input_ids, padding_tensor])
                    else:
                        item["input_ids"] = input_ids + [self.pad_token_id] * padding_length
                    
                    # Добавляем attention_mask если требуется
                    if "attention_mask" in item:
                        mask = item["attention_mask"]
                        if isinstance(mask, int):
                            item["attention_mask"] = [mask] + [0] * padding_length
                        elif hasattr(mask, 'shape'):
                            padding_mask = torch.zeros(padding_length, dtype=mask.dtype)
                            item["attention_mask"] = torch.cat([mask, padding_mask])
                        else:
                            item["attention_mask"] = mask + [0] * padding_length
                    elif return_attention_mask:
                        if isinstance(input_ids, int):
                            item["attention_mask"] = [1] + [0] * padding_length
                        elif hasattr(input_ids, 'shape'):
                            attention_mask = torch.ones(current_len, dtype=torch.long)
                            padding_mask = torch.zeros(padding_length, dtype=torch.long)
                            item["attention_mask"] = torch.cat([attention_mask, padding_mask])
                        else:
                            item["attention_mask"] = [1] * current_len + [0] * padding_length
        
        # Конвертируем в тензоры если требуется
        if return_tensors == "pt":
            import torch
            for key in list(encoded_inputs[0].keys()):
                if isinstance(encoded_inputs[0][key], list):
                    for i in range(len(encoded_inputs)):
                        encoded_inputs[i][key] = torch.tensor(encoded_inputs[i][key])
        
        return encoded_inputs
    
    def get_vocab(self) -> Dict[str, int]:
        """Возвращает словарь токенизатора."""
        return self._vocab
    
    def __len__(self) -> int:
        """Возвращает размер словаря."""
        return self.vocab_size
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Сохраняет токенизатор в формате HuggingFace.
        
        Args:
            save_directory: Директория для сохранения
            **kwargs: Дополнительные параметры
        """
        import os
        
        # Создаем директорию если не существует
        os.makedirs(save_directory, exist_ok=True)
        
        # Сохраняем конфигурацию токенизатора
        tokenizer_config = {
            "tokenizer_class": self.__class__.__name__,
            "llm_tokenizer_type": self.llm_tokenizer.__class__.__name__,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        # Сохраняем словарь
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Токенизатор сохранен в {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Загружает адаптированный токенизатор.
        
        Args:
            pretrained_model_name_or_path: Путь к сохраненному токенизатору
            **kwargs: Дополнительные параметры
            
        Returns:
            HFTokenizerAdapter: Загруженный адаптер
        """
        import os
        
        # Проверяем, является ли путь директорией с файлами токенизатора
        if os.path.isdir(pretrained_model_name_or_path):
            # Загружаем из директории
            config_path = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
            vocab_path = os.path.join(pretrained_model_name_or_path, "vocab.json")
            
            if not os.path.exists(config_path) or not os.path.exists(vocab_path):
                raise FileNotFoundError(
                    f"Файлы токенизатора не найдены в {pretrained_model_name_or_path}"
                )
            
            # Загружаем конфигурацию
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Определяем тип токенизатора llm
            llm_tokenizer_type = config.get("llm_tokenizer_type", "BPETokenizer")
            
            if llm_tokenizer_type == "BPETokenizer":
                # Создаем BPETokenizer и загружаем словарь
                llm_tokenizer = BPETokenizer()
                
                # Загружаем словарь
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                
                llm_tokenizer.vocab = vocab
                llm_tokenizer.inverse_vocab = {v: k for k, v in vocab.items()}
                llm_tokenizer.vocab_size = len(vocab)
                
                # Устанавливаем специальные токены
                llm_tokenizer.pad_token = config.get("pad_token", "<pad>")
                llm_tokenizer.unk_token = config.get("unk_token", "<unk>")
                llm_tokenizer.bos_token = config.get("bos_token", "<bos>")
                llm_tokenizer.eos_token = config.get("eos_token", "<eos>")
                
                llm_tokenizer.pad_token_id = config.get("pad_token_id", 0)
                llm_tokenizer.unk_token_id = config.get("unk_token_id", 1)
                llm_tokenizer.bos_token_id = config.get("bos_token_id", 2)
                llm_tokenizer.eos_token_id = config.get("eos_token_id", 3)
                
                return cls(llm_tokenizer, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый тип токенизатора: {llm_tokenizer_type}")
        
        else:
            # Пытаемся загрузить как файл llm токенизатора
            try:
                llm_tokenizer = BPETokenizer.load(pretrained_model_name_or_path)
                return cls(llm_tokenizer, **kwargs)
            except:
                raise ValueError(
                    f"Не удалось загрузить токенизатор из {pretrained_model_name_or_path}"
                )


def create_hf_tokenizer(llm_tokenizer: BaseTokenizer) -> HFTokenizerAdapter:
    """
    Создает адаптер HuggingFace для кастомного токенизатора.
    
    Args:
        llm_tokenizer: Токенизатор из библиотеки llm
        
    Returns:
        HFTokenizerAdapter: Адаптированный токенизатор
    """
    return HFTokenizerAdapter(llm_tokenizer)


def convert_to_hf_format(llm_tokenizer: BaseTokenizer, save_directory: str):
    """
    Конвертирует кастомный токенизатор в формат HuggingFace.
    
    Args:
        llm_tokenizer: Токенизатор из llm
        save_directory: Директория для сохранения
    """
    adapter = create_hf_tokenizer(llm_tokenizer)
    adapter.save_pretrained(save_directory)
    return adapter
