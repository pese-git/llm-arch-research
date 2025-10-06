#!/usr/bin/env python3
"""
Experiment: generate_gpt_bpe.py
Description: Генерация текста обученной GPT моделью с BPE токенизатором.
Использует только библиотеку llm без зависимостей от HuggingFace.
"""

import torch
import os
import sys

# Добавляем путь к shared модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.models.gpt import GPT2
from llm.tokenizers import BPETokenizer

from shared.configs import BASE_GPT_CONFIG, TEST_PROMPTS, GENERATION_CONFIG, PATHS
from shared.data import print_experiment_info, ensure_directories, ExperimentLogger


def load_model_and_tokenizer() -> tuple:
    """
    Загружает обученную модель и токенизатор.

    Returns:
        tuple: (модель, токенизатор, конфигурация)
    """
    # Проверяем существование файлов
    if not os.path.exists(PATHS["gpt_bpe_model"]):
        raise FileNotFoundError(
            f"Модель не найдена: {PATHS['gpt_bpe_model']}\n"
            f"Сначала обучите модель: uv run python experiments/llm_only/train_gpt_bpe.py"
        )

    if not os.path.exists(PATHS["bpe_tokenizer"]):
        raise FileNotFoundError(f"Токенизатор не найден: {PATHS['bpe_tokenizer']}")

    # Загружаем конфигурацию модели
    import json

    with open(PATHS["gpt_bpe_config"], "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # Загружаем токенизатор
    print("🔧 Загрузка BPE токенизатора...")
    tokenizer = BPETokenizer.load(PATHS["bpe_tokenizer"])
    print(f"✅ Токенизатор загружен (vocab_size={tokenizer.get_vocab_size()})")

    # Загружаем модель
    print("🔧 Загрузка GPT2 модели...")
    model = GPT2(model_config)
    model.load_state_dict(torch.load(PATHS["gpt_bpe_model"], map_location="cpu"))
    model.eval()
    print("✅ Модель загружена")

    return model, tokenizer, model_config


def generate_text(
    model: GPT2, tokenizer: BPETokenizer, prompt: str, config: dict
) -> str:
    """
    Генерирует текст на основе промпта.

    Args:
        model: Обученная GPT модель
        tokenizer: BPE токенизатор
        prompt: Входной текст
        config: Конфигурация генерации

    Returns:
        str: Сгенерированный текст
    """
    print(f"🔤 Промпт: '{prompt}'")
    print(
        f"📊 Параметры: max_tokens={config['max_new_tokens']}, "
        f"temp={config['temperature']}, sample={config['do_sample']}"
    )

    # Кодируем промпт
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    print(f"🎯 Токены промпта: {input_ids}")
    print(f"🎯 Токены (текст): {tokenizer.tokenize(prompt)}")
    print("🔄 Генерация...")

    # Генерируем текст
    with torch.no_grad():
        generated_ids = model.generate(
            x=input_tensor,
            max_new_tokens=config["max_new_tokens"],
            do_sample=config["do_sample"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
        )

    # Декодируем результат
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text


def test_different_strategies(model: GPT2, tokenizer: BPETokenizer, prompt: str):
    """
    Тестирует разные стратегии генерации на одном промпте.

    Args:
        model: Обученная модель
        tokenizer: BPE токенизатор
        prompt: Тестовый промпт
    """
    print(f"\n🎭 Сравнение стратегий генерации для промпта: '{prompt}'")
    print("=" * 60)

    strategies = [
        {"name": "🎯 Жадный поиск", "do_sample": False, "temperature": 1.0},
        {"name": "🎲 Вероятностная (temp=0.7)", "do_sample": True, "temperature": 0.7},
        {"name": "🔥 Случайная (temp=1.2)", "do_sample": True, "temperature": 1.2},
        {
            "name": "❄️  Детерминированная (temp=0.3)",
            "do_sample": True,
            "temperature": 0.3,
        },
    ]

    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        try:
            config = GENERATION_CONFIG.copy()
            config.update(
                {
                    "do_sample": strategy["do_sample"],
                    "temperature": strategy["temperature"],
                    "max_new_tokens": 20,
                }
            )

            generated = generate_text(model, tokenizer, prompt, config)

            # Выделяем сгенерированную часть
            generated_part = generated[len(prompt) :]
            print(f"   📤 Промпт: '{prompt}'")
            print(f"   🎯 Сгенерировано: '{generated_part}'")
            print(f"   📄 Полный текст: '{generated}'")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")


def analyze_tokenization(tokenizer: BPETokenizer, texts: list):
    """
    Анализирует токенизацию различных текстов.

    Args:
        tokenizer: BPE токенизатор
        texts: Список текстов для анализа
    """
    print(f"\n🔍 Анализ токенизации BPE:")
    print("=" * 50)

    for i, text in enumerate(texts):
        print(f"\nТекст {i+1}: '{text}'")

        # Токенизация
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = tokenizer.tokenize(text)

        print(f"   Токены (ID): {tokens}")
        print(f"   Токены (текст): {token_strings}")
        print(f"   Количество токенов: {len(tokens)}")
        print(f"   Эффективность: {len(text)} символов → {len(tokens)} токенов")

        # Декодирование обратно
        decoded = tokenizer.decode(tokens)
        if text == decoded:
            print(f"   ✅ Декодирование корректно")
        else:
            print(f"   ⚠️  Расхождения: '{decoded}'")


def interactive_generation(model: GPT2, tokenizer: BPETokenizer):
    """
    Режим интерактивной генерации.

    Args:
        model: Обученная модель
        tokenizer: BPE токенизатор
    """
    print(f"\n💬 Интерактивная генерация (для выхода введите 'exit')")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n🔤 Введите промпт: ").strip()

            if user_input.lower() in ["exit", "quit", "выход"]:
                break

            if not user_input:
                continue

            # Запрашиваем параметры
            try:
                max_tokens = int(input("📏 Макс. токенов [50]: ") or "50")
                temperature = float(input("🌡️  Температура [0.7]: ") or "0.7")
                do_sample_input = input("🎲 Сэмплирование (y/n) [y]: ").lower()
                do_sample = do_sample_input != "n"
            except:
                max_tokens = 50
                temperature = 0.7
                do_sample = True
                print("⚠️  Использую параметры по умолчанию")

            config = GENERATION_CONFIG.copy()
            config.update(
                {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                }
            )

            generated = generate_text(model, tokenizer, user_input, config)

            generated_part = generated[len(user_input) :]
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
    experiment_name = "Генерация текста GPT2 + BPE (только llm)"
    experiment_config = {
        "model": "GPT2 с BPE токенизатором",
        "стратегия": "автономная генерация",
        "вход": "промпты",
        "выход": "сгенерированный текст",
    }

    print_experiment_info(experiment_name, experiment_config)
    ensure_directories()
    logger = ExperimentLogger(experiment_name)

    try:
        # Загружаем модель и токенизатор
        model, tokenizer, model_config = load_model_and_tokenizer()

        # === Анализ токенизации ===
        analysis_texts = [
            "Искусственный интеллект",
            "Нейронные сети",
            "Машинное обучение",
        ]
        analyze_tokenization(tokenizer, analysis_texts)

        # === Генерация с разными промптами ===
        print(f"\n🎯 Генерация текста с разными промптами")
        print("=" * 60)

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\n📝 Пример {i+1}/{len(TEST_PROMPTS)}")
            print("-" * 40)

            try:
                generated = generate_text(model, tokenizer, prompt, GENERATION_CONFIG)

                # Выделяем сгенерированную часть
                generated_part = generated[len(prompt) :]

                print(f"📤 Промпт: '{prompt}'")
                print(f"🎯 Сгенерировано: '{generated_part}'")
                print(f"📄 Полный текст: '{generated}'")
                print(f"📏 Длина: {len(generated)} символов")

                # Логируем успешную генерацию
                logger.log_metric(f"generation_length_{i}", len(generated))

            except Exception as e:
                print(f"❌ Ошибка при генерации: {e}")
                continue

        # === Сравнение стратегий генерации ===
        test_prompt = "Искусственный"
        test_different_strategies(model, tokenizer, test_prompt)

        # === Интерактивная генерация ===
        interactive_generation(model, tokenizer)

        # === Сохранение результатов ===
        logger.save_logs("checkpoints/llm_only_generation_logs.json")

        print(f"\n🎉 Эксперимент генерации завершен успешно!")

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Ошибка в эксперименте: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
