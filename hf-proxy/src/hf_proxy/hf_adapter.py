"""
Адаптер для интеграции моделей llm с HuggingFace Transformers.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import (
    PreTrainedModel, 
    GPT2LMHeadModel,
    GPT2Config,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .hf_config import HFAdapterConfig, HFPretrainedConfig
from llm.models.gpt import GPT


class HFGPTAdapter(PreTrainedModel):
    """
    Адаптер для модели GPT из библиотеки llm.
    Позволяет использовать кастомные GPT модели с HuggingFace Transformers.
    """
    config_class = HFPretrainedConfig
    
    def __init__(self, config: HFPretrainedConfig, llm_model: Optional[GPT] = None):
        """
        Инициализация адаптера.
        
        Args:
            config: Конфигурация HuggingFace
            llm_model: Опционально, предварительно созданная модель llm
        """
        super().__init__(config)
        
        # Преобразуем HF конфигурацию в формат llm
        llm_config = self._hf_to_llm_config(config)
        
        # Создаем или используем переданную модель
        if llm_model is None:
            self.llm_model = GPT(llm_config)
        else:
            self.llm_model = llm_model
        
        # Устанавливаем веса если они есть в конфигурации
        if hasattr(config, 'state_dict') and config.state_dict is not None:
            self.llm_model.load_state_dict(config.state_dict)
    
    def _hf_to_llm_config(self, hf_config: HFPretrainedConfig) -> dict:
        """
        Преобразует конфигурацию HF в формат llm.
        
        Args:
            hf_config: Конфигурация HuggingFace
            
        Returns:
            dict: Конфигурация для llm модели
        """
        return {
            "vocab_size": hf_config.vocab_size,
            "embed_dim": hf_config.hidden_size,
            "num_heads": hf_config.num_attention_heads,
            "num_layers": hf_config.num_hidden_layers,
            "max_position_embeddings": hf_config.max_position_embeddings,
            "dropout": hf_config.hidden_dropout_prob,
        }
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Прямой проход модели.
        
        Args:
            input_ids: Входные токены [batch_size, seq_len]
            attention_mask: Маска внимания [batch_size, seq_len]
            labels: Метки для вычисления loss [batch_size, seq_len]
            past_key_values: Кешированные ключи и значения
            use_cache: Использовать кеширование
            output_attentions: Возвращать веса внимания
            output_hidden_states: Возвращать скрытые состояния
            return_dict: Возвращать словарь вместо кортежа
            
        Returns:
            CausalLMOutputWithCrossAttentions или кортеж
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Основной forward pass
        logits = self.llm_model(input_ids)
        
        loss = None
        if labels is not None:
            # Сдвигаем логиты и метки для языкового моделирования
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Вычисляем cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,  # Наша модель пока не поддерживает кеширование
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.Tensor, 
        past_key_values: Optional[Tuple] = None,
        **kwargs
    ) -> dict:
        """
        Подготавливает входные данные для генерации.
        
        Args:
            input_ids: Входные токены
            past_key_values: Кешированные ключи и значения
            
        Returns:
            dict: Подготовленные входные данные
        """
        # Наша простая реализация пока не поддерживает past_key_values
        return {"input_ids": input_ids}
    
    def can_generate(self) -> bool:
        """Проверяет, может ли модель генерировать текст."""
        return True
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Генерация текста с поддержкой HuggingFace интерфейса.
        
        Args:
            input_ids: Входные токены
            attention_mask: Маска внимания
            generation_config: Конфигурация генерации
            logits_processor: Процессоры логитов
            stopping_criteria: Критерии остановки
            
        Returns:
            torch.Tensor: Сгенерированные токены
        """
        # Извлекаем обязательные параметры из kwargs или используем значения по умолчанию
        max_new_tokens = kwargs.pop('max_new_tokens', 50)
        do_sample = kwargs.pop('do_sample', True)
        
        # Используем встроенную генерацию llm модели
        return self.llm_model.generate(
            x=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            attention_mask=attention_mask,
            **kwargs
        )


class HFAdapter:
    """
    Основной класс адаптера для преобразования моделей llm в формат HuggingFace.
    """
    
    @staticmethod
    def from_llm_model(
        llm_model: GPT, 
        hf_config: Optional[HFAdapterConfig] = None
    ) -> HFGPTAdapter:
        """
        Создает адаптер из существующей llm модели.
        
        Args:
            llm_model: Обученная модель из библиотеки llm
            hf_config: Конфигурация для HuggingFace
            
        Returns:
            HFGPTAdapter: Адаптированная модель
        """
        if hf_config is None:
            # Создаем конфигурацию из модели llm
            hf_config = HFAdapterConfig.from_llm_config(llm_model.config)
        
        # Преобразуем в PretrainedConfig
        pretrained_config = HFPretrainedConfig(**hf_config.to_dict())
        
        return HFGPTAdapter(pretrained_config, llm_model)
    
    @staticmethod
    def from_pretrained(
        model_path: str,
        hf_config: Optional[HFAdapterConfig] = None
    ) -> HFGPTAdapter:
        """
        Загружает модель из чекпоинта и создает адаптер.
        
        Args:
            model_path: Путь к сохраненной модели
            hf_config: Конфигурация для HuggingFace
            
        Returns:
            HFGPTAdapter: Адаптированная модель
        """
        # Загружаем состояние модели
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Определяем конфигурацию из состояния модели или используем переданную
        if hf_config is None:
            # Пытаемся определить конфигурацию из состояния модели
            # Это упрощенный подход - в реальности нужно сохранять конфигурацию отдельно
            vocab_size = state_dict.get('_token_embeddings._embedding.weight', torch.zeros(50257, 768)).shape[0]
            embed_dim = state_dict.get('_token_embeddings._embedding.weight', torch.zeros(50257, 768)).shape[1]
            
            hf_config = HFAdapterConfig(
                vocab_size=vocab_size,
                hidden_size=embed_dim,
                # Остальные параметры можно установить по умолчанию
            )
        
        pretrained_config = HFPretrainedConfig(**hf_config.to_dict())
        
        # Создаем модель llm и загружаем веса
        llm_config = {
            "vocab_size": hf_config.vocab_size,
            "embed_dim": hf_config.hidden_size,
            "num_heads": hf_config.num_attention_heads,
            "num_layers": hf_config.num_hidden_layers,
            "max_position_embeddings": hf_config.max_position_embeddings,
            "dropout": hf_config.hidden_dropout_prob,
        }
        
        llm_model = GPT(llm_config)
        llm_model.load_state_dict(state_dict)
        
        return HFGPTAdapter(pretrained_config, llm_model)
    
    @staticmethod
    def save_pretrained(
        model: HFGPTAdapter,
        save_directory: str,
        **kwargs
    ):
        """
        Сохраняет адаптированную модель в формате HuggingFace.
        
        Args:
            model: Адаптированная модель
            save_directory: Директория для сохранения
            **kwargs: Дополнительные параметры
        """
        import os
        import json
        
        # Создаем директорию если не существует
        os.makedirs(save_directory, exist_ok=True)
        
        # Сохраняем конфигурацию
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model.config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Сохраняем веса модели
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model.llm_model.state_dict(), model_path)
        
        # Сохраняем токенизатор если передан
        if hasattr(kwargs, 'tokenizer') and kwargs['tokenizer'] is not None:
            kwargs['tokenizer'].save_pretrained(save_directory)
