"""
Утилиты для работы с адаптером HuggingFace.
"""

import torch
import json
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoConfig
from .hf_config import HFAdapterConfig, HFPretrainedConfig
from .hf_adapter import HFAdapter, HFGPTAdapter


class HFUtils:
    """
    Утилиты для работы с HuggingFace адаптером.
    """
    
    @staticmethod
    def create_hf_config_from_llm(llm_config: Dict[str, Any]) -> HFPretrainedConfig:
        """
        Создает конфигурацию HuggingFace из конфигурации llm.
        
        Args:
            llm_config: Конфигурация модели из библиотеки llm
            
        Returns:
            HFPretrainedConfig: Конфигурация для HuggingFace
        """
        adapter_config = HFAdapterConfig.from_llm_config(llm_config)
        return HFPretrainedConfig(**adapter_config.to_dict())
    
    @staticmethod
    def convert_to_hf_format(
        llm_model,
        tokenizer = None,
        model_name: str = "custom-gpt"
    ) -> tuple:
        """
        Конвертирует llm модель в формат HuggingFace.
        
        Args:
            llm_model: Модель из библиотеки llm
            tokenizer: Токенизатор (HF или кастомный)
            model_name: Имя модели для сохранения
            
        Returns:
            tuple: (адаптированная модель, токенизатор)
        """
        # Создаем адаптер
        hf_model = HFAdapter.from_llm_model(llm_model)
        
        # Если токенизатор не передан, создаем стандартный
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # Устанавливаем специальные токены
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(tokenizer, '__class__') and 'BPETokenizer' in str(tokenizer.__class__):
            # Если передан наш кастомный токенизатор, создаем адаптер
            from .hf_tokenizer import create_hf_tokenizer
            tokenizer = create_hf_tokenizer(tokenizer)
        
        return hf_model, tokenizer
    
    @staticmethod
    def push_to_hub(
        model: HFGPTAdapter,
        tokenizer,
        repo_name: str,
        organization: Optional[str] = None,
        private: bool = False,
        **kwargs
    ):
        """
        Загружает модель в HuggingFace Hub.
        
        Args:
            model: Адаптированная модель
            tokenizer: Токенизатор
            repo_name: Имя репозитория
            organization: Организация (опционально)
            private: Приватный репозиторий
            **kwargs: Дополнительные параметры
        """
        try:
            from huggingface_hub import HfApi, ModelCard, create_repo
            
            # Создаем репозиторий
            if organization:
                repo_id = f"{organization}/{repo_name}"
            else:
                repo_id = repo_name
            
            create_repo(repo_id, private=private, exist_ok=True)
            
            # Сохраняем модель локально
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Сохраняем модель
                HFAdapter.save_pretrained(model, tmp_dir, tokenizer=tokenizer)
                
                # Создаем Model Card
                card = ModelCard.from_template(
                    model_name=repo_name,
                    language="ru",
                    license="apache-2.0",
                    tags=["llm", "gpt", "custom"],
                )
                card.save(os.path.join(tmp_dir, "README.md"))
                
                # Загружаем в Hub
                api = HfApi()
                api.upload_folder(
                    folder_path=tmp_dir,
                    repo_id=repo_id,
                    commit_message="Initial commit with custom GPT model"
                )
                
            print(f"✅ Модель успешно загружена в HuggingFace Hub: {repo_id}")
            
        except ImportError:
            raise ImportError(
                "Для загрузки в HuggingFace Hub установите huggingface_hub: "
                "pip install huggingface_hub"
            )
    
    @staticmethod
    def load_from_hub(
        repo_id: str,
        **kwargs
    ) -> tuple:
        """
        Загружает модель из HuggingFace Hub.
        
        Args:
            repo_id: ID репозитория
            **kwargs: Дополнительные параметры
            
        Returns:
            tuple: (модель, токенизатор)
        """
        from transformers import AutoTokenizer
        
        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
        
        # Загружаем конфигурацию
        config = AutoConfig.from_pretrained(repo_id, **kwargs)
        
        # Создаем модель llm на основе конфигурации
        llm_config = {
            "vocab_size": config.vocab_size,
            "embed_dim": config.hidden_size,
            "num_heads": config.num_attention_heads,
            "num_layers": config.num_hidden_layers,
            "max_position_embeddings": config.max_position_embeddings,
            "dropout": config.hidden_dropout_prob,
        }
        
        # Загружаем модель через адаптер
        model = HFAdapter.from_pretrained(
            f"{repo_id}/pytorch_model.bin",
            HFAdapterConfig.from_llm_config(llm_config)
        )
        
        return model, tokenizer
    
    @staticmethod
    def compare_with_hf_model(
        llm_model,
        hf_model_name: str = "gpt2",
        test_input: str = "Hello world"
    ) -> Dict[str, Any]:
        """
        Сравнивает llm модель с эталонной моделью из HuggingFace.
        
        Args:
            llm_model: Модель из библиотеки llm
            hf_model_name: Имя модели HuggingFace для сравнения
            test_input: Тестовый вход
            
        Returns:
            Dict: Результаты сравнения
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Загружаем эталонную модель
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        
        # Подготавливаем входные данные
        inputs = hf_tokenizer(test_input, return_tensors="pt")
        
        # Получаем логиты от обеих моделей
        with torch.no_grad():
            hf_logits = hf_model(**inputs).logits
            llm_logits = llm_model(inputs['input_ids'])
        
        # Сравниваем результаты
        hf_probs = torch.softmax(hf_logits[0, -1], dim=-1)
        llm_probs = torch.softmax(llm_logits[0, -1], dim=-1)
        
        # Вычисляем метрики
        kl_divergence = torch.nn.functional.kl_div(
            torch.log(llm_probs + 1e-8),
            hf_probs,
            reduction='batchmean'
        )
        
        cosine_similarity = torch.nn.functional.cosine_similarity(
            hf_logits.flatten(),
            llm_logits.flatten(),
            dim=0
        )
        
        return {
            "kl_divergence": kl_divergence.item(),
            "cosine_similarity": cosine_similarity.item(),
            "hf_top_tokens": torch.topk(hf_probs, 5).indices.tolist(),
            "llm_top_tokens": torch.topk(llm_probs, 5).indices.tolist(),
        }


class TokenizerWrapper:
    """
    Обертка для токенизатора с дополнительными утилитами.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode_batch(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Кодирует батч текстов.
        
        Args:
            texts: Список текстов
            **kwargs: Дополнительные параметры токенизации
            
        Returns:
            Dict: Токенизированные данные
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        )
    
    def decode_batch(self, token_ids: torch.Tensor, **kwargs) -> List[str]:
        """
        Декодирует батч токенов.
        
        Args:
            token_ids: Тензор с токенами
            **kwargs: Дополнительные параметры декодирования
            
        Returns:
            List[str]: Декодированные тексты
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        texts = []
        for i in range(token_ids.size(0)):
            text = self.tokenizer.decode(
                token_ids[i],
                skip_special_tokens=True,
                **kwargs
            )
            texts.append(text)
        
        return texts
    
    def get_vocab_size(self) -> int:
        """Возвращает размер словаря."""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Возвращает специальные токены."""
        return {
            "pad_token": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token_id,
            "unk_token": self.tokenizer.unk_token_id,
        }


def create_hf_pipeline(
    llm_model,
    tokenizer=None,
    device: str = "auto",
    **kwargs
):
    """
    Создает HuggingFace pipeline из llm модели.
    
    Args:
        llm_model: Модель из библиотеки llm
        tokenizer: Токенизатор
        device: Устройство для вычислений
        **kwargs: Дополнительные параметры pipeline
        
    Returns:
        transformers.Pipeline: Готовый pipeline
    """
    from transformers import pipeline
    
    # Конвертируем модель в HF формат
    hf_model, tokenizer = HFUtils.convert_to_hf_format(llm_model, tokenizer)
    
    # Создаем pipeline
    pipe = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        device=device,
        **kwargs
    )
    
    return pipe
