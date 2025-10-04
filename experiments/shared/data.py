"""
Общие утилиты для работы с данными в экспериментах.
"""

import os
from typing import List, Tuple
from .configs import TRAIN_TEXTS, PATHS


def load_training_data(split_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Загружает данные для обучения и разделяет на train/validation.
    
    Args:
        split_ratio: Доля данных для обучения
        
    Returns:
        Tuple: (train_texts, val_texts)
    """
    train_size = int(len(TRAIN_TEXTS) * split_ratio)
    train_data = TRAIN_TEXTS[:train_size]
    val_data = TRAIN_TEXTS[train_size:]
    
    return train_data, val_data


def ensure_directories():
    """Создает необходимые директории если они не существуют."""
    directories = [
        "checkpoints",
        "checkpoints/gpt-bpe", 
        "checkpoints/hf-bpe-tokenizer",
        "checkpoints/hf-trained",
        "checkpoints/hf-trained-proxy",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_model_paths(experiment_type: str = "llm_only") -> dict:
    """
    Возвращает пути для конкретного типа эксперимента.
    
    Args:
        experiment_type: Тип эксперимента ('llm_only' или 'hf_integration')
        
    Returns:
        dict: Словарь с путями
    """
    base_paths = PATHS.copy()
    
    if experiment_type == "hf_integration":
        base_paths.update({
            "model": base_paths["hf_model"],
            "tokenizer": base_paths["hf_tokenizer"]
        })
    else:  # llm_only
        base_paths.update({
            "model": base_paths["gpt_bpe_model"],
            "tokenizer": base_paths["bpe_tokenizer"]
        })
    
    return base_paths


def print_experiment_info(experiment_name: str, config: dict):
    """
    Выводит информацию о запускаемом эксперименте.
    
    Args:
        experiment_name: Название эксперимента
        config: Конфигурация эксперимента
    """
    print("=" * 70)
    print(f"🚀 Эксперимент: {experiment_name}")
    print("=" * 70)
    print("📊 Конфигурация:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()


def save_experiment_results(results: dict, filepath: str):
    """
    Сохраняет результаты эксперимента в файл.
    
    Args:
        results: Словарь с результатами
        filepath: Путь для сохранения
    """
    import json
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Результаты эксперимента сохранены: {filepath}")


def load_experiment_results(filepath: str) -> dict:
    """
    Загружает результаты эксперимента из файла.
    
    Args:
        filepath: Путь к файлу с результатами
        
    Returns:
        dict: Загруженные результаты
    """
    import json
    
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class ExperimentLogger:
    """
    Логгер для экспериментов.
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = {}
    
    def log_metric(self, name: str, value: float):
        """Логирует метрику."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        print(f"📈 {name}: {value:.4f}")
    
    def log_step(self, step: int, loss: float, **kwargs):
        """Логирует шаг обучения."""
        print(f"📊 Step {step}: loss={loss:.4f}", end="")
        for key, value in kwargs.items():
            print(f", {key}={value:.4f}", end="")
        print()
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None):
        """Логирует завершение эпохи."""
        print(f"🎯 Epoch {epoch}: train_loss={train_loss:.4f}", end="")
        if val_loss is not None:
            print(f", val_loss={val_loss:.4f}", end="")
        print()
    
    def save_logs(self, filepath: str):
        """Сохраняет логи эксперимента."""
        import json
        
        logs = {
            "experiment_name": self.experiment_name,
            "metrics": self.metrics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Логи эксперимента сохранены: {filepath}")
