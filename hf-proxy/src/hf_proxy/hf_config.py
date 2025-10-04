"""
Конфигурационные классы для адаптации моделей llm к HuggingFace.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from transformers import PretrainedConfig

@dataclass
class HFAdapterConfig:
    """
    Конфигурация для адаптера HuggingFace.
    
    Параметры:
        model_type: Тип модели (gpt, llama, etc.)
        vocab_size: Размер словаря
        hidden_size: Размер скрытого слоя
        num_hidden_layers: Количество слоев
        num_attention_heads: Количество голов внимания
        max_position_embeddings: Максимальная длина последовательности
        intermediate_size: Размер промежуточного слоя FFN
        hidden_dropout_prob: Вероятность dropout
        attention_probs_dropout_prob: Вероятность dropout в внимании
        initializer_range: Диапазон инициализации весов
        layer_norm_eps: Эпсилон для LayerNorm
        use_cache: Использовать кеширование
        pad_token_id: ID токена паддинга
        eos_token_id: ID токена конца строки
        bos_token_id: ID токена начала строки
    """
    model_type: str = "gpt"
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 1024
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    bos_token_id: int = 50256
    
    # Дополнительные параметры для совместимости
    architectures: list = field(default_factory=lambda: ["GPT2LMHeadModel"])
    torch_dtype: str = "float32"
    transformers_version: str = "4.44.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
    
    @classmethod
    def from_llm_config(cls, llm_config: Dict[str, Any]) -> "HFAdapterConfig":
        """
        Создает конфигурацию HF из конфигурации llm.
        
        Args:
            llm_config: Конфигурация модели из библиотеки llm
            
        Returns:
            HFAdapterConfig: Конфигурация для HuggingFace
        """
        # Маппинг параметров из llm в HF формат
        mapping = {
            "embed_dim": "hidden_size",
            "num_layers": "num_hidden_layers", 
            "num_heads": "num_attention_heads",
            "max_position_embeddings": "max_position_embeddings",
            "dropout": "hidden_dropout_prob",
            "vocab_size": "vocab_size"
        }
        
        hf_config_dict = {}
        for llm_key, hf_key in mapping.items():
            if llm_key in llm_config:
                hf_config_dict[hf_key] = llm_config[llm_key]
        
        # Устанавливаем промежуточный размер (обычно 4x hidden_size)
        if "hidden_size" in hf_config_dict:
            hf_config_dict["intermediate_size"] = hf_config_dict["hidden_size"] * 4
        
        return cls(**hf_config_dict)


class HFPretrainedConfig(PretrainedConfig):
    """
    Конфигурация для предобученных моделей HuggingFace.
    Наследуется от PretrainedConfig для полной совместимости.
    """
    model_type = "gpt"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=1024,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=50256,
        eos_token_id=50256,
        bos_token_id=50256,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
