#!/usr/bin/env python3
"""
Experiment: train_with_hf_trainer.py
Description: Обучение GPT модели через HuggingFace Trainer с использованием hf-proxy.
Интегрирует кастомную модель llm с инструментами HuggingFace.
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
    TRAINING_CONFIG,
    PATHS,
    TEST_PROMPTS,
)
from shared.data import (
    load_training_data,
    ensure_directories,
    print_experiment_info,
    ExperimentLogger,
)


def setup_hf_training():
    """
    Настраивает окружение для обучения через HuggingFace Trainer.

    Returns:
        tuple: (hf_model, hf_tokenizer, llm_tokenizer, model_config)
    """
    print("🔧 Настройка HuggingFace обучения...")

    # === Подготовка данных ===
    train_texts, val_texts = load_training_data()
    print(f"📊 Данные: {len(train_texts)} train, {len(val_texts)} validation")

    # === Обучение/загрузка токенизатора ===
    if os.path.exists(PATHS["bpe_tokenizer"]):
        print("📝 Загрузка BPE токенизатора...")
        llm_tokenizer = BPETokenizer.load(PATHS["bpe_tokenizer"])
        print(f"✅ Токенизатор загружен (vocab_size={llm_tokenizer.get_vocab_size()})")
    else:
        print("📝 Обучение BPE токенизатора...")
        llm_tokenizer = BPETokenizer()
        llm_tokenizer.train(
            texts=TRAIN_TEXTS,
            vocab_size=BPE_CONFIG["vocab_size"],
            special_tokens=BPE_CONFIG["special_tokens"],
        )
        llm_tokenizer.save(PATHS["bpe_tokenizer"])
        print(f"✅ Токенизатор обучен и сохранен")

    # === Создание адаптера токенизатора ===
    print("🔧 Создание адаптера HuggingFace для токенизатора...")
    hf_tokenizer = HFTokenizerAdapter(llm_tokenizer)
    print(f"✅ Адаптер токенизатора создан")

    # === Инициализация модели ===
    model_config = BASE_GPT_CONFIG.copy()
    model_config["vocab_size"] = llm_tokenizer.get_vocab_size()

    print("🔧 Создание GPT модели...")
    llm_model = GPT(model_config)

    # === Создание адаптера модели ===
    print("🔧 Создание адаптера HuggingFace для модели...")
    hf_model = HFAdapter.from_llm_model(llm_model)
    print(f"✅ Адаптер модели создан")

    return hf_model, hf_tokenizer, llm_tokenizer, model_config, train_texts, val_texts


def test_hf_integration(hf_model, hf_tokenizer, llm_tokenizer):
    """
    Тестирует интеграцию с HuggingFace инструментами.

    Args:
        hf_model: Адаптированная модель
        hf_tokenizer: Адаптированный токенизатор
        llm_tokenizer: Оригинальный токенизатор
    """
    print("\n🧪 Тестирование интеграции с HuggingFace...")

    test_texts = ["Искусственный интеллект", "Нейронные сети"]

    for text in test_texts:
        print(f"\n🔤 Текст: '{text}'")

        # Тестируем адаптированный токенизатор
        hf_inputs = hf_tokenizer(text, return_tensors="pt")
        print(f"   HF токенизатор: {hf_inputs['input_ids'].shape}")

        # Тестируем оригинальный токенизатор для сравнения
        original_tokens = llm_tokenizer.encode(text)
        print(f"   Оригинальный токенизатор: {len(original_tokens)} токенов")

        # Тестируем forward pass через адаптированную модель
        try:
            with torch.no_grad():
                outputs = hf_model(**hf_inputs)
            print(f"   HF forward pass: успешно (logits: {outputs.logits.shape})")
        except Exception as e:
            print(f"   ❌ HF forward pass: {e}")


def main():
    """Основная функция эксперимента."""
    # === Настройка эксперимента ===
    experiment_name = "Обучение GPT через HF Trainer (с hf-proxy)"
    experiment_config = {
        "model": "GPT через HFAdapter",
        "tokenizer": "BPE через HFTokenizerAdapter",
        "trainer": "HuggingFace Trainer",
        "vocab_size": BPE_CONFIG["vocab_size"],
        "training_epochs": TRAINING_CONFIG["num_epochs"],
    }

    print_experiment_info(experiment_name, experiment_config)
    ensure_directories()
    logger = ExperimentLogger(experiment_name)

    try:
        # Настраиваем обучение
        (
            hf_model,
            hf_tokenizer,
            llm_tokenizer,
            model_config,
            train_texts,
            val_texts,
        ) = setup_hf_training()

        # Тестируем интеграцию
        test_hf_integration(hf_model, hf_tokenizer, llm_tokenizer)

        # === Подготовка датасетов HuggingFace ===
        print(f"\n📊 Подготовка датасетов HuggingFace...")

        from datasets import Dataset

        def tokenize_function(examples):
            """Функция токенизации для HF datasets."""
            # Используем адаптированный токенизатор
            tokenized = hf_tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=model_config["max_position_embeddings"],
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Создаем датасеты
        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})

        # Токенизируем
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        print(f"   Train датасет: {len(train_dataset)} примеров")
        print(f"   Validation датасет: {len(val_dataset)} примеров")

        # === Настройка HuggingFace Trainer ===
        print(f"\n🔧 Настройка HuggingFace Trainer...")

        from transformers import (
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )

        # Data collator для языкового моделирования
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=hf_tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # Аргументы обучения
        training_args = TrainingArguments(
            output_dir=PATHS["hf_model"],
            overwrite_output_dir=True,
            num_train_epochs=TRAINING_CONFIG["num_epochs"],
            per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            logging_dir="./logs",
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            report_to=None,
        )

        # Создаем Trainer
        trainer = Trainer(
            model=hf_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        print("✅ HuggingFace Trainer настроен")

        # === Запуск обучения ===
        print(f"\n🎯 Запуск обучения через HuggingFace Trainer...")

        train_result = trainer.train()

        # Сохраняем лучшую модель
        trainer.save_model()
        hf_tokenizer.save_pretrained(PATHS["hf_model"])

        print("✅ Обучение завершено успешно!")
        print(f"📊 Final train loss: {train_result.metrics['train_loss']:.4f}")

        if "eval_loss" in train_result.metrics:
            print(f"📊 Final eval loss: {train_result.metrics['eval_loss']:.4f}")

        # === Сохранение через hf-proxy ===
        print(f"\n💾 Сохранение через hf-proxy...")

        from hf_proxy import convert_to_hf_format

        # Сохраняем токенизатор в HF формате
        hf_tokenizer_dir = PATHS["hf_tokenizer"]
        hf_tokenizer.save_pretrained(hf_tokenizer_dir)

        # Сохраняем модель через hf-proxy
        hf_proxy_dir = PATHS["hf_proxy_model"]
        HFAdapter.save_pretrained(hf_model, hf_proxy_dir, tokenizer=hf_tokenizer)

        print(f"✅ Модель сохранена в HF формате:")
        print(f"   - {PATHS['hf_model']}: стандартный HF формат")
        print(f"   - {hf_proxy_dir}: через hf-proxy")
        print(f"   - {hf_tokenizer_dir}: токенизатор в HF формате")

        # === Тестирование генерации ===
        print(f"\n🧪 Тестирование генерации после обучения...")
        hf_model.eval()

        for prompt in TEST_PROMPTS[:3]:
            print(f"\n🔤 Промпт: '{prompt}'")

            try:
                inputs = hf_tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    generated = hf_model.generate(
                        # Trainer мог перенести модель на GPU/MPS
                        input_ids=inputs["input_ids"].to(hf_model.device),
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.8,
                    )

                generated_text = hf_tokenizer.decode(
                    generated[0], skip_special_tokens=True
                )
                print(f"🎯 Результат: '{generated_text}'")

            except Exception as e:
                print(f"❌ Ошибка генерации: {e}")

        # === Сохранение результатов ===
        results = {
            "experiment": experiment_name,
            "model_config": model_config,
            "training_config": TRAINING_CONFIG,
            "final_loss": train_result.metrics.get("train_loss", "N/A"),
            "eval_loss": train_result.metrics.get("eval_loss", "N/A"),
        }

        logger.save_logs("checkpoints/hf_integration_training_logs.json")

        print(f"\n🎉 Эксперимент с HF интеграцией завершен успешно!")
        print(f"\n💡 Для использования обученной модели:")
        print(f"   uv run python experiments/hf_integration/generate_with_hf_tools.py")

    except Exception as e:
        print(f"❌ Ошибка в эксперименте: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
