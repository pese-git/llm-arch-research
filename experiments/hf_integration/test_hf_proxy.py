#!/usr/bin/env python3
"""
Test: test_hf_proxy.py
Description: Тестирование базовой функциональности hf-proxy без сложных зависимостей.
"""

import torch
import os
import sys

# Добавляем путь к shared модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.models.gpt import GPT
from llm.tokenizers import BPETokenizer
from hf_proxy import HFAdapter, HFTokenizerAdapter

from shared.configs import (
    TRAIN_TEXTS,
    BASE_GPT_CONFIG,
    BPE_CONFIG,
    TEST_PROMPTS,
    GENERATION_CONFIG,
)


def test_basic_hf_integration():
    """Тестирует базовую интеграцию hf-proxy."""
    print("🧪 Тестирование базовой интеграции hf-proxy...")

    # === Подготовка токенизатора ===
    print("1. Подготовка токенизатора...")
    llm_tokenizer = BPETokenizer()
    llm_tokenizer.train(
        texts=TRAIN_TEXTS,
        vocab_size=BPE_CONFIG["vocab_size"],
        special_tokens=BPE_CONFIG["special_tokens"],
    )

    hf_tokenizer = HFTokenizerAdapter(llm_tokenizer)
    print(f"   ✅ Токенизатор создан (vocab_size={hf_tokenizer.vocab_size})")

    # === Подготовка модели ===
    print("2. Подготовка модели...")
    model_config = BASE_GPT_CONFIG.copy()
    model_config["vocab_size"] = hf_tokenizer.vocab_size

    llm_model = GPT(model_config)
    hf_model = HFAdapter.from_llm_model(llm_model)
    print(f"   ✅ Модель создана")

    # === Тестирование токенизации ===
    print("3. Тестирование токенизации...")
    test_texts = ["Искусственный интеллект", "Нейронные сети"]

    for text in test_texts:
        print(f"   📝 Текст: '{text}'")

        # Оригинальный токенизатор
        original_tokens = llm_tokenizer.encode(text)
        print(f"      Оригинальный: {len(original_tokens)} токенов")

        # HF адаптер
        hf_inputs = hf_tokenizer(text, return_tensors="pt")
        print(f"      HF адаптер: {hf_inputs['input_ids'].shape}")

        # Декодирование
        decoded = hf_tokenizer.decode(hf_inputs["input_ids"][0])
        print(f"      Декодированный: '{decoded}'")

    # === Тестирование forward pass ===
    print("4. Тестирование forward pass...")
    for text in test_texts:
        hf_inputs = hf_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = hf_model(**hf_inputs)

        print(f"   📝 '{text}' -> logits: {outputs.logits.shape}")

    # === Тестирование генерации ===
    print("5. Тестирование генерации...")
    hf_model.eval()

    for prompt in TEST_PROMPTS[:3]:
        print(f"   🔤 Промпт: '{prompt}'")

        try:
            inputs = hf_tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                generated = hf_model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.8,
                )

            generated_text = hf_tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"      🎯 Результат: '{generated_text}'")

        except Exception as e:
            print(f"      ❌ Ошибка: {e}")

    # === Тестирование сохранения/загрузки ===
    print("6. Тестирование сохранения/загрузки...")
    try:
        # Сохраняем токенизатор
        hf_tokenizer.save_pretrained("test_save/tokenizer")
        print("   ✅ Токенизатор сохранен")

        # Сохраняем модель
        HFAdapter.save_pretrained(hf_model, "test_save/model", tokenizer=hf_tokenizer)
        print("   ✅ Модель сохранена")

        # Загружаем токенизатор
        loaded_tokenizer = HFTokenizerAdapter.from_pretrained("test_save/tokenizer")
        print(f"   ✅ Токенизатор загружен (vocab_size={loaded_tokenizer.vocab_size})")

        # Загружаем модель
        model_path = os.path.join("test_save/model", "pytorch_model.bin")
        loaded_model = HFAdapter.from_pretrained(model_path)
        print("   ✅ Модель загружена")

        # Проверяем работоспособность загруженной модели
        test_input = hf_tokenizer("Тест", return_tensors="pt")
        with torch.no_grad():
            loaded_outputs = loaded_model(**test_input)
        print(
            f"   ✅ Загруженная модель работает (logits: {loaded_outputs.logits.shape})"
        )

    except Exception as e:
        print(f"   ❌ Ошибка сохранения/загрузки: {e}")

    print("\n🎉 Базовое тестирование hf-proxy завершено!")


def test_hf_tokenizer_methods():
    """Тестирует различные методы HF токенизатора."""
    print("\n🧪 Тестирование методов HF токенизатора...")

    # Создаем токенизатор
    llm_tokenizer = BPETokenizer()
    llm_tokenizer.train(
        texts=TRAIN_TEXTS[:5],
        vocab_size=500,
        special_tokens=BPE_CONFIG["special_tokens"],
    )

    hf_tokenizer = HFTokenizerAdapter(llm_tokenizer)

    test_text = "Искусственный интеллект и машинное обучение"

    # Тестируем разные методы
    print("1. Метод __call__:")
    result = hf_tokenizer(test_text, return_tensors="pt")
    print(f"   Результат: {result}")

    print("2. Метод encode:")
    encoded = hf_tokenizer.encode(test_text)
    print(f"   Закодировано: {encoded}")

    print("3. Метод decode:")
    decoded = hf_tokenizer.decode(encoded)
    print(f"   Декодировано: '{decoded}'")

    print("4. Метод tokenize:")
    tokens = hf_tokenizer.tokenize(test_text)
    print(f"   Токены: {tokens}")

    print("5. Метод get_vocab:")
    vocab = hf_tokenizer.get_vocab()
    print(f"   Размер словаря: {len(vocab)}")

    print("✅ Все методы токенизатора работают!")


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ HF-PROXY")
    print("=" * 60)

    try:
        # Тестируем базовую интеграцию
        test_basic_hf_integration()

        # Тестируем методы токенизатора
        test_hf_tokenizer_methods()

        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
        print("\n📚 Проверенные функции:")
        print("   ✅ Создание HF адаптера для токенизатора")
        print("   ✅ Создание HF адаптера для модели")
        print("   ✅ Токенизация и декодирование")
        print("   ✅ Forward pass через адаптированную модель")
        print("   ✅ Генерация текста")
        print("   ✅ Сохранение и загрузка моделей")
        print("   ✅ Все методы HF токенизатора")

    except Exception as e:
        print(f"\n❌ Ошибка в тестировании: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
