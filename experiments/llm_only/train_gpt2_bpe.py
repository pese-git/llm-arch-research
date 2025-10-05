#!/usr/bin/env python3
"""
Experiment: train_gpt_bpe.py
Description: Обучение GPT модели с собственным BPE токенизатором.
Использует только библиотеку llm без зависимостей от HuggingFace.
"""

import torch
import os
import sys

# Добавляем путь к shared модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.models.gpt import GPT2
from llm.tokenizers import BPETokenizer
from llm.training.dataset import TextDataset
from llm.training.trainer import Trainer

from shared.configs import (
    TRAIN_TEXTS, BASE_GPT_CONFIG, BPE_CONFIG, 
    TRAINING_CONFIG, PATHS, TEST_PROMPTS
)
from shared.data import (
    load_training_data, ensure_directories, 
    print_experiment_info, ExperimentLogger
)


def train_bpe_tokenizer(texts: list, config: dict) -> BPETokenizer:
    """
    Обучает BPE токенизатор на текстах.
    
    Args:
        texts: Список текстов для обучения
        config: Конфигурация токенизатора
        
    Returns:
        BPETokenizer: Обученный токенизатор
    """
    print("🔧 Обучение BPE токенизатора...")
    
    tokenizer = BPETokenizer()
    tokenizer.train(
        texts=texts,
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"]
    )
    
    # Сохраняем токенизатор
    os.makedirs(os.path.dirname(PATHS["bpe_tokenizer"]), exist_ok=True)
    tokenizer.save(PATHS["bpe_tokenizer"])
    
    print(f"✅ BPE токенизатор обучен и сохранен: {PATHS['bpe_tokenizer']}")
    print(f"📊 Размер словаря: {tokenizer.get_vocab_size()}")
    
    return tokenizer


def test_tokenizer(tokenizer: BPETokenizer, texts: list):
    """
    Тестирует токенизатор на примерах.
    
    Args:
        tokenizer: Обученный токенизатор
        texts: Список тестовых текстов
    """
    print("\n🧪 Тестирование токенизатора:")
    
    for i, text in enumerate(texts[:3]):
        print(f"\nПример {i+1}:")
        print(f"   Исходный текст: '{text}'")
        
        # Кодирование
        tokens = tokenizer.encode(text)
        token_strings = tokenizer.tokenize(text)
        
        print(f"   Токены (ID): {tokens}")
        print(f"   Токены (текст): {token_strings}")
        print(f"   Количество токенов: {len(tokens)}")
        
        # Декодирование
        decoded = tokenizer.decode(tokens)
        print(f"   Декодированный: '{decoded}'")
        
        if text == decoded:
            print("   ✅ Кодирование/декодирование корректно")
        else:
            print("   ⚠️  Небольшие расхождения")


def main():
    """Основная функция эксперимента."""
    # === Настройка эксперимента ===
    experiment_name = "Обучение GPT2 с BPE токенизатором (только llm)"
    experiment_config = {
        "model": "GPT2",
        "tokenizer": "BPE", 
        "vocab_size": BPE_CONFIG["vocab_size"],
        "training_epochs": TRAINING_CONFIG["num_epochs"],
        "batch_size": TRAINING_CONFIG["batch_size"],
        "learning_rate": TRAINING_CONFIG["learning_rate"]
    }
    
    print_experiment_info(experiment_name, experiment_config)
    ensure_directories()
    logger = ExperimentLogger(experiment_name)
    
    try:
        # === Подготовка данных ===
        train_texts, val_texts = load_training_data()
        print(f"📊 Данные: {len(train_texts)} train, {len(val_texts)} validation")
        
        # === Обучение токенизатора ===
        if os.path.exists(PATHS["bpe_tokenizer"]):
            print("📝 Загрузка предварительно обученного токенизатора...")
            tokenizer = BPETokenizer.load(PATHS["bpe_tokenizer"])
            print(f"✅ Токенизатор загружен (vocab_size={tokenizer.get_vocab_size()})")
        else:
            tokenizer = train_bpe_tokenizer(TRAIN_TEXTS, BPE_CONFIG)
        
        # Тестируем токенизатор
        test_tokenizer(tokenizer, TEST_PROMPTS[:3])
        
        # === Инициализация модели ===
        model_config = BASE_GPT_CONFIG.copy()
        model_config["vocab_size"] = tokenizer.get_vocab_size()
        
        print(f"\n🔧 Инициализация GPT2 модели...")
        print(f"   Размер словаря: {model_config['vocab_size']}")
        print(f"   Размер эмбеддингов: {model_config['embed_dim']}")
        print(f"   Количество слоев: {model_config['num_layers']}")
        print(f"   Количество голов внимания: {model_config['num_heads']}")
        
        model = GPT2(model_config)
        
        # === Подготовка датасета ===
        print(f"\n📊 Подготовка датасета...")
        train_dataset = TextDataset(
            train_texts, 
            tokenizer, 
            block_size=model_config["max_position_embeddings"]
        )
        print(f"   Размер train датасета: {len(train_dataset)} примеров")
        
        # === Обучение модели ===
        print(f"\n🎯 Начало обучения GPT2 модели...")
        
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            lr=TRAINING_CONFIG["learning_rate"],
            batch_size=TRAINING_CONFIG["batch_size"],
            num_epochs=TRAINING_CONFIG["num_epochs"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"]
        )
        
        # Запускаем обучение
        trainer.train()
        
        # === Сохранение модели ===
        print(f"\n💾 Сохранение модели...")
        os.makedirs(os.path.dirname(PATHS["gpt_bpe_model"]), exist_ok=True)
        
        # Сохраняем модель
        torch.save(model.state_dict(), PATHS["gpt_bpe_model"])
        
        # Сохраняем конфигурацию
        import json
        with open(PATHS["gpt_bpe_config"], 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Модель сохранена:")
        print(f"   - {PATHS['gpt_bpe_model']}: веса модели")
        print(f"   - {PATHS['gpt_bpe_config']}: конфигурация модели")
        print(f"   - {PATHS['bpe_tokenizer']}: токенизатор")
        
        # === Тестирование генерации ===
        print(f"\n🧪 Тестирование генерации текста...")
        model.eval()
        
        for prompt in TEST_PROMPTS[:3]:
            print(f"\n🔤 Промпт: '{prompt}'")
            
            try:
                # Кодируем промпт
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                input_tensor = torch.tensor([input_ids], dtype=torch.long)
                
                # Генерируем текст
                with torch.no_grad():
                    generated_ids = model.generate(
                        x=input_tensor,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.8
                    )
                
                # Декодируем результат
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                generated_part = generated_text[len(prompt):]
                
                print(f"🎯 Сгенерировано: '{generated_part}'")
                print(f"📄 Полный текст: '{generated_text}'")
                
            except Exception as e:
                print(f"❌ Ошибка генерации: {e}")
        
        # === Сохранение результатов ===
        results = {
            "experiment": experiment_name,
            "model_config": model_config,
            "training_config": TRAINING_CONFIG,
            "tokenizer_vocab_size": tokenizer.get_vocab_size(),
            "final_loss": "см. логи обучения"  # В реальном эксперименте можно сохранить final loss
        }
        
        logger.save_logs("checkpoints/llm_only_training_logs.json")
        
        print(f"\n🎉 Эксперимент завершен успешно!")
        print(f"\n💡 Для использования обученной модели:")
        print(f"   uv run python experiments/llm_only/generate_gpt_bpe.py")
        
    except Exception as e:
        print(f"❌ Ошибка в эксперименте: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
