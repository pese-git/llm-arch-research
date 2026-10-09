#!/usr/bin/env python3
"""
Бенчмарк архитектур: обучение всех шести моделей на одном корпусе в одинаковых условиях.

    uv run python experiments/llm_only/benchmark.py --data-dir data/novels --steps 1000

Что одинаково у всех моделей:
- корпус и блоки (TokenBlockDataset из data-dir, подготовленный prepare_corpus.py), токенизатор;
- общие размеры: embed_dim, num_layers, число голов, контекст, dropout;
- обучение: число шагов, батч, learning rate, warmup, seed. Порядок батчей задаётся seed и
  номером эпохи, поэтому все модели видят одни и те же батчи в одном порядке;
- начальные веса моделей определяются torch.manual_seed(seed) перед созданием.

Что подбирается под архитектуру, чтобы общее число параметров было сопоставимым (≈5 млн):
- intermediate_size у gated-FFN (LLaMA, Mistral, Gemma): 688 ≈ 8/3·d вместо 4·d — как в LLaMA,
  чтобы три матрицы SwiGLU/GeGLU весили столько же, сколько две матрицы GELU-FFN у GPT;
- число и размер экспертов Mixtral: 8 экспертов с intermediate_size 96, top-2.

Результаты — в <out>/results.json и <out>/results.md: параметры (всего и активных на токен),
валидационный loss по шагам, итоговая перплексия на всём val, время и токенов в секунду.
"""

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.datasets.token_block_dataset import TokenBlockDataset
from llm.evaluation import perplexity
from llm.tokenizers import BPETokenizer
from llm.training.trainer import Trainer

from run_llm_experiment import load_model_class  # noqa: E402  (тот же реестр моделей, что у скрипта)

MODELS = ["gpt", "gpt2", "llama", "mistral", "mixtral", "gemma"]
GATED_FFN = 688  # ≈ 8/3 · 256, кратно 16


def model_config(name, vocab_size, embed_dim=256, num_layers=4, num_heads=4, block_size=128,
                 dropout=0.1, window_size=32, router_aux_loss_coef=0.01):
    """Конфиг модели: общие размеры плюс то, что нужно архитектуре, чтобы параметров было поровну."""
    common = {"vocab_size": vocab_size, "embed_dim": embed_dim, "num_layers": num_layers,
              "max_position_embeddings": block_size, "dropout": dropout}
    head_size = embed_dim // num_heads
    if name in ("gpt", "gpt2"):
        return {**common, "num_heads": num_heads}
    if name == "llama":
        return {**common, "num_heads": num_heads, "intermediate_size": GATED_FFN}
    if name == "mistral":
        return {**common, "num_q_heads": num_heads, "num_kv_heads": num_heads // 2,
                "head_size": head_size, "window_size": window_size, "intermediate_size": GATED_FFN}
    if name == "mixtral":
        return {**common, "num_q_heads": num_heads, "num_kv_heads": num_heads // 2,
                "head_size": head_size, "num_experts": 8, "top_k_experts": 2,
                "intermediate_size": 96, "router_aux_loss_coef": router_aux_loss_coef}
    if name == "gemma":
        return {**common, "num_q_heads": num_heads, "head_size": head_size, "intermediate_size": GATED_FFN}
    raise ValueError(f"Модель '{name}' не поддерживается")


def count_parameters(model, config):
    """Всего параметров и активных на токен (у Mixtral из экспертов работают top_k из num_experts)."""
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for n, p in model.named_parameters() if "_experts" in n)
    if experts:
        active = total - experts * (1 - config["top_k_experts"] / config["num_experts"])
    else:
        active = total
    return total, int(active)


def run_one(name, args, tokenizer, train, val, device):
    config = model_config(name, tokenizer.get_vocab_size(), embed_dim=args.embed_dim,
                          num_layers=args.num_layers, num_heads=args.num_heads,
                          block_size=args.block_size, dropout=args.dropout)
    torch.manual_seed(args.seed)
    model = load_model_class(name)(config)
    total, active = count_parameters(model, config)

    # Прогрев: первые шаги на MPS и CUDA компилируют ядра и искажали бы время первой модели.
    # Прогреваем копию, настоящая модель остаётся с начальными весами.
    warm = Trainer(copy.deepcopy(model), train, batch_size=args.batch_size, device=device,
                   max_steps=3, warmup_steps=0)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warm.train()
    del warm

    trainer = Trainer(model, train, val, lr=args.lr, batch_size=args.batch_size, device=device,
                      max_steps=args.steps, warmup_ratio=args.warmup_ratio,
                      eval_interval=args.eval_interval, seed=args.seed)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    val_ppl = perplexity(model, DataLoader(val, batch_size=args.batch_size), device=trainer.device)
    log = trainer.state.log
    tokens = args.steps * args.batch_size * args.block_size
    result = {
        "model": name,
        "params": total,
        "active_params": active,
        "config": config,
        "final_val_loss": log[-1]["val_loss"],
        "best_val_loss": min(r["val_loss"] for r in log),
        "val_perplexity": val_ppl,
        "final_train_loss": log[-1]["train_loss"],
        "val_curve": [(r["step"], r["val_loss"]) for r in log],
        "train_seconds": train_seconds,
        "tokens_per_second": tokens / train_seconds,
        "device": str(trainer.device),
    }
    return result


def markdown_table(results, args):
    lines = [
        f"Корпус `{args.data_dir}`, {args.steps} шагов × {args.batch_size} блоков × {args.block_size} токенов, "
        f"lr {args.lr}, warmup {args.warmup_ratio:.0%}, seed {args.seed}, устройство {results[0]['device']}.",
        "",
        "| Модель | Параметры | Активных на токен | Val loss (лучший) | Val loss (финал) | Перплексия | Время, с | Токенов/с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['params'] / 1e6:.2f}M | {r['active_params'] / 1e6:.2f}M | "
            f"{r['best_val_loss']:.3f} | {r['final_val_loss']:.3f} | {r['val_perplexity']:.1f} | "
            f"{r['train_seconds']:.0f} | {r['tokens_per_second']:.0f} |"
        )
    steps = [s for s, _ in results[0]["val_curve"]]
    lines += ["", "Валидационный loss по шагам:", "",
              "| Модель | " + " | ".join(str(s) for s in steps) + " |",
              "|---|" + "---|" * len(steps)]
    for r in results:
        lines.append(f"| {r['model']} | " + " | ".join(f"{v:.3f}" for _, v in r["val_curve"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк: обучение всех моделей в одинаковых условиях")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--data-dir", default="data/novels", help="каталог с train.bin, val.bin, tokenizer.json")
    parser.add_argument("--out", default="checkpoints/benchmark")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(os.path.join(args.data_dir, "tokenizer.json"))
    train = TokenBlockDataset(os.path.join(args.data_dir, "train.bin"), args.block_size)
    val = TokenBlockDataset(os.path.join(args.data_dir, "val.bin"), args.block_size)
    print(f"Корпус: {len(train)} train-блоков, {len(val)} val-блоков, словарь {tokenizer.get_vocab_size()}")

    os.makedirs(args.out, exist_ok=True)
    results = []
    for name in args.models:
        print(f"\n===== {name} =====", flush=True)
        results.append(run_one(name, args, tokenizer, train, val, args.device))
        with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
        r = results[-1]
        print(f"[{name}] params {r['params'] / 1e6:.2f}M, val ppl {r['val_perplexity']:.1f}, "
              f"{r['train_seconds']:.0f} s, {r['tokens_per_second']:.0f} tok/s", flush=True)

    table = markdown_table(results, args)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
