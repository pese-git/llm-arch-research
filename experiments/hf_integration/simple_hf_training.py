#!/usr/bin/env python3
"""
Experiment: simple_hf_training.py
Description: Упрощенное обучение GPT модели с использованием hf-proxy.
Использует ручное обучение вместо сложного HuggingFace Trainer.
"""

import torch
import torch.nn as nn
import os
import sys
import json

# Добавляем путь к shared модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.models.gpt import GPT
from llm.tokenizers import BPETokenizer
from hf_proxy import HFAdapter, HFTokenizerAdapter

from shared.configs import (
    TRAIN_TEXTS,
    BASE_GPT_CONFIG,
    BPE_CONFIG,
    TRAINING_CONFIG,
    PATHS,
    TEST_PROMPTS,
)


def create_dataset(hf_tokenizer, texts, max_length=128):
    """
    Создает простой датасет для обучения.

    Args:
        hf_tokenizer: Адаптированный токенизатор
        texts: Список текстов
        max_length: Максимальная длина последовательности

    Returns:
        list: Список тензоров input_ids
    """
    dataset = []

    for text in texts:
        # Токенизируем текст
        inputs = hf_tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]

        # Создаем метки для языкового моделирования
        labels = input_ids.clone()

        dataset.append({"input_ids": input_ids, "labels": labels})

    return dataset


def manual_training_loop(hf_model, hf_tokenizer, train_texts, val_texts, config):
    """
    Ручной цикл обучения без использования Trainer.

    Args:
        hf_model: Адаптированная модель
        hf_tokenizer: Адаптированный токенизатор
        train_texts: Тексты для обучения
        val_texts: Тексты для валидации
        config: Конфигурация обучения

    Returns:
        dict: Результаты обучения
    """
    print("🎯 Запуск ручного обучения...")

    # Создаем датасеты
    train_dataset = create_dataset(hf_tokenizer, train_texts)
    val_dataset = create_dataset(hf_tokenizer, val_texts)

    print(f"📊 Данные: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Оптимизатор
    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=config["learning_rate"])

    # Функция потерь
    loss_fn = nn.CrossEntropyLoss()

    # Обучение
    hf_model.train()
    train_losses = []
    val_losses = []

    for epoch in range(config["num_epochs"]):
        print(f"\n📅 Эпоха {epoch + 1}/{config['num_epochs']}")

        # Обучение
        epoch_train_loss = 0
        for i, batch in enumerate(train_dataset):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].unsqueeze(0)  # [1, seq_len]
            labels = batch["labels"].unsqueeze(0)  # [1, seq_len]

            # Forward pass
            outputs = hf_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if i % 5 == 0:
                print(f"   Batch {i}/{len(train_dataset)}: loss = {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        print(f"   📊 Средняя train loss: {avg_train_loss:.4f}")

        # Валидация
        hf_model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_dataset:
                input_ids = batch["input_ids"].unsqueeze(0)
                labels = batch["labels"].unsqueeze(0)

                outputs = hf_model(input_ids=input_ids, labels=labels)
                epoch_val_loss += outputs.loss.item()

        avg_val_loss = epoch_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        print(f"   📊 Средняя val loss: {avg_val_loss:.4f}")

        hf_model.train()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }


def test_generation_after_training(hf_model, hf_tokenizer, test_prompts):
    """
    Тестирует генерацию после обучения.

    Args:
        hf_model: Обученная модель
        hf_tokenizer: Токенизатор
        test_prompts: Тестовые промпты
    """
    print("\n🧪 Тестирование генерации после обучения...")
    hf_model.eval()

    for prompt in test_prompts[:3]:
        print(f"\n🔤 Промпт: '{prompt}'")

        try:
            inputs = hf_tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                generated = hf_model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.8,
                )

            generated_text = hf_tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"🎯 Результат: '{generated_text}'")

        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")


def main():
    """Основная функция эксперимента."""
    print("=" * 60)
    print("🚀 УПРОЩЕННОЕ ОБУЧЕНИЕ GPT С HF-PROXY")
    print("=" * 60)

    try:
        # === Подготовка данных ===
        print("🔧 Подготовка данных...")
        train_texts = TRAIN_TEXTS[
            :10
        ]  # Используем меньше данных для быстрого тестирования
        val_texts = TRAIN_TEXTS[10:12]

        print(f"📊 Данные: {len(train_texts)} train, {len(val_texts)} validation")

        # === Подготовка токенизатора ===
        print("🔧 Подготовка токенизатора...")
        llm_tokenizer = BPETokenizer()
        llm_tokenizer.train(
            texts=train_texts,
            vocab_size=BPE_CONFIG["vocab_size"],
            special_tokens=BPE_CONFIG["special_tokens"],
        )

        hf_tokenizer = HFTokenizerAdapter(llm_tokenizer)
        print(f"✅ Токенизатор создан (vocab_size={hf_tokenizer.vocab_size})")

        # === Подготовка модели ===
        print("🔧 Подготовка модели...")
        model_config = BASE_GPT_CONFIG.copy()
        model_config["vocab_size"] = hf_tokenizer.vocab_size

        llm_model = GPT(model_config)
        hf_model = HFAdapter.from_llm_model(llm_model)
        print(f"✅ Модель создана")

        # === Тестирование до обучения ===
        print("\n🧪 Тестирование до обучения...")
        test_generation_after_training(hf_model, hf_tokenizer, TEST_PROMPTS)

        # === Обучение ===
        print(f"\n🎯 Обучение модели...")
        training_config = {
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "num_epochs": 2,  # Меньше эпох для быстрого тестирования
            "batch_size": TRAINING_CONFIG["batch_size"],
        }

        results = manual_training_loop(
            hf_model, hf_tokenizer, train_texts, val_texts, training_config
        )

        print(f"\n📊 Результаты обучения:")
        print(f"   Final train loss: {results['final_train_loss']:.4f}")
        print(f"   Final val loss: {results['final_val_loss']:.4f}")

        # === Тестирование после обучения ===
        print("\n🧪 Тестирование после обучения...")
        test_generation_after_training(hf_model, hf_tokenizer, TEST_PROMPTS)

        # === Сохранение модели ===
        print(f"\n💾 Сохранение модели...")

        # Создаем директории
        os.makedirs("checkpoints/hf_simple_trained", exist_ok=True)
        os.makedirs("checkpoints/hf_simple_tokenizer", exist_ok=True)

        # Сохраняем токенизатор
        hf_tokenizer.save_pretrained("checkpoints/hf_simple_tokenizer")
        print("✅ Токенизатор сохранен")

        # Сохраняем модель
        HFAdapter.save_pretrained(
            hf_model, "checkpoints/hf_simple_trained", tokenizer=hf_tokenizer
        )
        print("✅ Модель сохранена")

        # Сохраняем результаты
        results_path = "checkpoints/simple_training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "training_config": training_config,
                    "model_config": model_config,
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"✅ Результаты сохранены в {results_path}")

        print(f"\n🎉 Упрощенное обучение завершено успешно!")
        print(f"\n💡 Для использования обученной модели:")
        print(f"   uv run python experiments/hf_integration/generate_with_hf_tools.py")

    except Exception as e:
        print(f"❌ Ошибка в эксперименте: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
