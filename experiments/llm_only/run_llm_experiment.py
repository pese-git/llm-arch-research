#!/usr/bin/env python3
"""
Универсальный скрипт для обучения и генерации LLM.
Позволяет выбирать тип модели и действие через аргументы,
а специальные параметры подавать отдельным JSON-конфигом.
"""

import argparse
import json
import os
import sys

import torch

# Добавляем директорию shared среди импортируемых
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.tokenizers import BPETokenizer
from llm.datasets.text_dataset import TextDataset
from llm.training.trainer import Trainer

from shared.data import (
    print_experiment_info,
    ensure_directories,
    load_training_data,
    ExperimentLogger,
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_class(model_name):
    if model_name.lower() == 'gpt':
        from llm.models.gpt import GPT
        return GPT
    elif model_name.lower() == 'gpt2':
        from llm.models.gpt import GPT2
        return GPT2
    elif model_name.lower() == 'llama':
        from llm.models.llama import Llama
        return Llama
    elif model_name.lower() == 'mistral':
        from llm.models.mistral import Mistral
        return Mistral
    elif model_name.lower() == 'mixtral':
        from llm.models.mixtral import Mixtral
        return Mixtral
    elif model_name.lower() == 'gemma':
        from llm.models.gemma import Gemma
        return Gemma
    else:
        raise ValueError(f"Модель '{model_name}' не поддерживается.")


def main():
    parser = argparse.ArgumentParser(description='Универсальный запуск обучения/генерации LLM.')
    parser.add_argument('--model', '-m', type=str, required=True, help='Название модели (gpt, gpt2, llama и т.д.).')
    parser.add_argument('--action', '-a', type=str, required=True, choices=['train', 'generate'], help='Действие: train или generate.')
    parser.add_argument('--config', '-c', type=str, required=True, help='Путь к JSON-конфигу с параметрами.')
    args = parser.parse_args()

    config = load_config(args.config)
    ModelClass = load_model_class(args.model)
    logger = ExperimentLogger(f"{args.action}_{args.model}")

    print_experiment_info(f"Эксперимент {args.action} {args.model}", config)
    ensure_directories()

    # ==== Обучение ====
    if args.action == 'train':
        train_texts, val_texts = load_training_data()
        # --- Токенизатор ---
        if os.path.exists(config["bpe_tokenizer"]):
            print("📝 Загрузка обученного токенизатора...")
            tokenizer = BPETokenizer.load(config["bpe_tokenizer"])
            print(f"✅ Токенизатор загружен (vocab_size={tokenizer.get_vocab_size()})")
        else:
            print("🔧 Обучение BPE токенизатора...")
            tokenizer = BPETokenizer()
            tokenizer.train(
                texts=train_texts,
                vocab_size=config["bpe_vocab_size"],
                special_tokens=config["bpe_special_tokens"]
            )
            os.makedirs(os.path.dirname(config["bpe_tokenizer"]), exist_ok=True)
            tokenizer.save(config["bpe_tokenizer"])
            print(f"✅ BPE токенизатор обучен и сохранен: {config['bpe_tokenizer']}")

        # Тестируем токенизатор (базово)
        for test_text in config.get("test_prompts", ["Тест"]):
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            print(f"[TEST TOK] '{test_text}' → {encoded} → '{decoded}'")

        # --- Модель ---
        model_config = config["model_config"]
        model_config["vocab_size"] = tokenizer.get_vocab_size()
        model = ModelClass(model_config)

        # --- Датасет ---
        train_dataset = TextDataset(
            train_texts,
            tokenizer,
            block_size=model_config["max_position_embeddings"]
        )
        print(f"   Размер train датасета: {len(train_dataset)} примеров")

        # --- Trainer ---
        training = config["training"]
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            lr=training["learning_rate"],
            batch_size=training["batch_size"],
            num_epochs=training["num_epochs"],
            warmup_steps=training.get("warmup_steps", 0),
        )
        trainer.train()

        # --- Сохранение модели ---
        os.makedirs(os.path.dirname(config["model_weights"]), exist_ok=True)
        torch.save(model.state_dict(), config["model_weights"])
        with open(config["model_config_path"], "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        print(f"✅ Модель сохранена: {config['model_weights']}")

        logger.save_logs(config.get("log_path", "checkpoints/llm_only_training_logs.json"))

    # ==== Генерация ====
    elif args.action == 'generate':
        # --- Загрузка ---
        if not os.path.exists(config["model_weights"]):
            raise FileNotFoundError(f"Модель не найдена: {config['model_weights']}")
        if not os.path.exists(config["bpe_tokenizer"]):
            raise FileNotFoundError(f"Токенизатор не найден: {config['bpe_tokenizer']}")
        with open(config["model_config_path"], "r", encoding="utf-8") as f:
            model_config = json.load(f)
        tokenizer = BPETokenizer.load(config["bpe_tokenizer"])
        model = ModelClass(model_config)
        model.load_state_dict(torch.load(config["model_weights"], map_location="cpu"))
        model.eval()

        def generate(prompt, gen_cfg):
            print(f"Промпт: {prompt}")
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            with torch.no_grad():
                generated_ids = model.generate(
                    x=input_tensor,
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    do_sample=gen_cfg["do_sample"],
                    temperature=gen_cfg["temperature"],
                    top_k=gen_cfg.get("top_k"),
                    top_p=gen_cfg.get("top_p"),
                )
            return tokenizer.decode(generated_ids[0].tolist())

        prompts = config.get("test_prompts", ["Тестовый промпт"])
        gen_cfg = config.get("generation", {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "do_sample": True,
            "top_k": None,
            "top_p": None
        })
        for prompt in prompts:
            generated = generate(prompt, gen_cfg)
            print(f"\n[RESULT] Prompt: '{prompt}'\n---\n{generated}\n{'='*60}")

        logger.save_logs(config.get("log_path", "checkpoints/llm_only_generation_logs.json"))

if __name__ == "__main__":
    main()
