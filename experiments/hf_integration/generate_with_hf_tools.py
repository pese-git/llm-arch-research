#!/usr/bin/env python3
"""
Experiment: generate_with_hf_tools.py
Description: Генерация текста обученной GPT моделью через HuggingFace инструменты.
Использует hf-proxy для интеграции кастомной модели с HF экосистемой.
"""

import torch
import os
import sys

# Добавляем путь к shared модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_proxy import HFAdapter, HFTokenizerAdapter, create_hf_pipeline

from shared.configs import (
    TEST_PROMPTS, GENERATION_CONFIG, PATHS
)
from shared.data import (
    print_experiment_info, ensure_directories, ExperimentLogger
)


def load_hf_model_and_tokenizer() -> tuple:
    """
    Загружает модель и токенизатор в формате HuggingFace.
    
    Returns:
        tuple: (hf_model, hf_tokenizer, model_config)
    """
    # Используем упрощенную версию модели
    model_path = "checkpoints/hf_simple_trained"
    tokenizer_path = "checkpoints/hf_simple_tokenizer"
    
    # Проверяем существование файлов
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}\n"
            f"Сначала обучите модель: uv run python experiments/hf_integration/simple_hf_training.py"
        )
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Токенизатор не найден: {tokenizer_path}"
        )
    
    # Загружаем адаптированный токенизатор
    print("🔧 Загрузка адаптированного токенизатора...")
    hf_tokenizer = HFTokenizerAdapter.from_pretrained(tokenizer_path)
    print(f"✅ Токенизатор загружен (vocab_size={hf_tokenizer.vocab_size})")
    
    # Загружаем конфигурацию модели
    import json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    
    # Загружаем модель через HFAdapter с правильной конфигурацией
    print("🔧 Загрузка адаптированной модели...")
    model_bin_path = os.path.join(model_path, "pytorch_model.bin")
    
    # Создаем конфигурацию из сохраненного config.json
    from hf_proxy import HFAdapterConfig
    hf_config = HFAdapterConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        max_position_embeddings=model_config["max_position_embeddings"],
        hidden_dropout_prob=model_config.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=model_config.get("attention_probs_dropout_prob", 0.1),
    )
    
    hf_model = HFAdapter.from_pretrained(model_bin_path, hf_config=hf_config)
    hf_model.eval()
    print("✅ Модель загружена")
    
    return hf_model, hf_tokenizer, model_config


def test_hf_pipeline(hf_model, hf_tokenizer):
    """
    Тестирует создание HuggingFace pipeline.
    
    Args:
        hf_model: Адаптированная модель
        hf_tokenizer: Адаптированный токенизатор
    """
    print("\n🧪 Тестирование HuggingFace pipeline...")
    
    try:
        # Создаем pipeline
        pipe = create_hf_pipeline(
            hf_model,
            tokenizer=hf_tokenizer,
            device="cpu",
            max_length=50,
            do_sample=True,
            temperature=0.7
        )
        
        print("✅ HuggingFace pipeline создан")
        
        # Тестируем pipeline
        test_prompts = TEST_PROMPTS[:3]
        
        for prompt in test_prompts:
            print(f"\n🔤 Промпт: '{prompt}'")
            
            try:
                result = pipe(prompt, max_new_tokens=20)
                print(f"🎯 Результат: {result[0]['generated_text']}")
            except Exception as e:
                print(f"❌ Ошибка в pipeline: {e}")
                
    except Exception as e:
        print(f"❌ Ошибка создания pipeline: {e}")


def generate_with_hf_model(hf_model, hf_tokenizer, prompt: str, config: dict) -> str:
    """
    Генерирует текст через адаптированную модель HF.
    
    Args:
        hf_model: Адаптированная модель
        hf_tokenizer: Адаптированный токенизатор
        prompt: Входной текст
        config: Конфигурация генерации
        
    Returns:
        str: Сгенерированный текст
    """
    print(f"🔤 Промпт: '{prompt}'")
    print(f"📊 Параметры: max_tokens={config['max_new_tokens']}, "
          f"temp={config['temperature']}, sample={config['do_sample']}")
    
    # Кодируем через адаптированный токенизатор
    inputs = hf_tokenizer(prompt, return_tensors="pt")
    
    print(f"🎯 Токены промпта: {inputs['input_ids'].tolist()[0]}")
    print("🔄 Генерация через HF адаптер...")
    
    # Генерируем через адаптированную модель
    with torch.no_grad():
        generated_ids = hf_model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=config["max_new_tokens"],
            do_sample=config["do_sample"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"]
        )
    
    # Декодируем через адаптированный токенизатор
    generated_text = hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def test_different_hf_strategies(hf_model, hf_tokenizer, prompt: str):
    """
    Тестирует разные стратегии генерации через HF интерфейс.
    
    Args:
        hf_model: Адаптированная модель
        hf_tokenizer: Адаптированный токенизатор
        prompt: Тестовый промпт
    """
    print(f"\n🎭 Сравнение стратегий генерации через HF для промпта: '{prompt}'")
    print("=" * 70)
    
    strategies = [
        {"name": "🎯 Жадный поиск", "do_sample": False, "temperature": 1.0},
        {"name": "🎲 Вероятностная (temp=0.7)", "do_sample": True, "temperature": 0.7},
        {"name": "🔥 Случайная (temp=1.2)", "do_sample": True, "temperature": 1.2},
        {"name": "❄️  Детерминированная (temp=0.3)", "do_sample": True, "temperature": 0.3},
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        try:
            config = GENERATION_CONFIG.copy()
            config.update({
                "do_sample": strategy["do_sample"],
                "temperature": strategy["temperature"],
                "max_new_tokens": 20
            })
            
            generated = generate_with_hf_model(hf_model, hf_tokenizer, prompt, config)
            
            # Выделяем сгенерированную часть
            generated_part = generated[len(prompt):]
            print(f"   📤 Промпт: '{prompt}'")
            print(f"   🎯 Сгенерировано: '{generated_part}'")
            print(f"   📄 Полный текст: '{generated}'")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")


def analyze_hf_tokenization(hf_tokenizer, texts: list):
    """
    Анализирует токенизацию через адаптированный токенизатор.
    
    Args:
        hf_tokenizer: Адаптированный токенизатор
        texts: Список текстов для анализа
    """
    print(f"\n🔍 Анализ токенизации через HF адаптер:")
    print("=" * 60)
    
    for i, text in enumerate(texts):
        print(f"\nТекст {i+1}: '{text}'")
        
        # Токенизация через адаптер
        inputs = hf_tokenizer(text, return_tensors="pt")
        tokens = inputs['input_ids'].tolist()[0]
        token_strings = hf_tokenizer.tokenize(text)
        
        print(f"   Токены (ID): {tokens}")
        print(f"   Токены (текст): {token_strings}")
        print(f"   Количество токенов: {len(tokens)}")
        
        # Декодирование обратно
        decoded = hf_tokenizer.decode(tokens)
        print(f"   Декодированный: '{decoded}'")
        
        if text == decoded:
            print(f"   ✅ Декодирование корректно")
        else:
            print(f"   ⚠️  Расхождения")


def interactive_hf_generation(hf_model, hf_tokenizer):
    """
    Режим интерактивной генерации через HF интерфейс.
    
    Args:
        hf_model: Адаптированная модель
        hf_tokenizer: Адаптированный токенизатор
    """
    print(f"\n💬 Интерактивная генерация через HF (для выхода введите 'exit')")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🔤 Введите промпт: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                break
                
            if not user_input:
                continue
            
            # Запрашиваем параметры
            try:
                max_tokens = int(input("📏 Макс. токенов [50]: ") or "50")
                temperature = float(input("🌡️  Температура [0.7]: ") or "0.7")
                do_sample_input = input("🎲 Сэмплирование (y/n) [y]: ").lower()
                do_sample = do_sample_input != 'n'
            except:
                max_tokens = 50
                temperature = 0.7
                do_sample = True
                print("⚠️  Использую параметры по умолчанию")
            
            config = GENERATION_CONFIG.copy()
            config.update({
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": do_sample
            })
            
            generated = generate_with_hf_model(hf_model, hf_tokenizer, user_input, config)
            
            generated_part = generated[len(user_input):]
            print(f"\n🎯 Результат:")
            print(f"   📤 Промпт: '{user_input}'")
            print(f"   🎯 Сгенерировано: '{generated_part}'")
            print(f"   📄 Полный текст: '{generated}'")
            
        except KeyboardInterrupt:
            print("\n👋 Завершение работы...")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def main():
    """Основная функция эксперимента."""
    # === Настройка эксперимента ===
    experiment_name = "Генерация текста через HF инструменты (с hf-proxy)"
    experiment_config = {
        "model": "GPT через HFAdapter",
        "tokenizer": "BPE через HFTokenizerAdapter",
        "инструменты": "HuggingFace pipeline & генерация",
        "стратегия": "интеграция с HF экосистемой"
    }
    
    print_experiment_info(experiment_name, experiment_config)
    ensure_directories()
    logger = ExperimentLogger(experiment_name)
    
    try:
        # Загружаем модель и токенизатор в HF формате
        hf_model, hf_tokenizer, model_config = load_hf_model_and_tokenizer()
        
        # === Анализ токенизации ===
        analysis_texts = [
            "Искусственный интеллект",
            "Нейронные сети", 
            "Машинное обучение"
        ]
        analyze_hf_tokenization(hf_tokenizer, analysis_texts)
        
        # === Тестирование HF pipeline ===
        test_hf_pipeline(hf_model, hf_tokenizer)
        
        # === Генерация с разными промптами ===
        print(f"\n🎯 Генерация текста через HF адаптер")
        print("=" * 60)
        
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n📝 Пример {i+1}/{len(TEST_PROMPTS)}")
            print("-" * 40)
            
            try:
                generated = generate_with_hf_model(hf_model, hf_tokenizer, prompt, GENERATION_CONFIG)
                
                # Выделяем сгенерированную часть
                generated_part = generated[len(prompt):]
                
                print(f"📤 Промпт: '{prompt}'")
                print(f"🎯 Сгенерировано: '{generated_part}'")
                print(f"📄 Полный текст: '{generated}'")
                print(f"📏 Длина: {len(generated)} символов")
                
                # Логируем успешную генерацию
                logger.log_metric(f"hf_generation_length_{i}", len(generated))
                
            except Exception as e:
                print(f"❌ Ошибка при генерации: {e}")
                continue
        
        # === Сравнение стратегий генерации ===
        test_prompt = "Искусственный"
        test_different_hf_strategies(hf_model, hf_tokenizer, test_prompt)
        
        # === Интерактивная генерация ===
        interactive_hf_generation(hf_model, hf_tokenizer)
        
        # === Сохранение результатов ===
        logger.save_logs("checkpoints/hf_integration_generation_logs.json")
        
        print(f"\n🎉 Эксперимент с HF интеграцией завершен успешно!")
        print(f"\n📚 Достигнутая интеграция:")
        print(f"   ✅ Загрузка модели и токенизатора в HF формате")
        print(f"   ✅ Использование HF pipeline")
        print(f"   ✅ Генерация через стандартные HF интерфейсы")
        print(f"   ✅ Совместимость с HF экосистемой")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Ошибка в эксперименте: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
