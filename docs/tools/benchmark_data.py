"""Собирает docs/assets/benchmark/results.json из результатов experiments/llm_only/benchmark.py.

Каждый запуск benchmark.py с одним seed пишет results.json; здесь они объединяются в один
компактный файл, из которого docs/tools/figures.py строит рисунки benchmark-*:

    uv run python docs/tools/benchmark_data.py checkpoints/benchmark/results.json \\
        checkpoints/benchmark/seed1/results.json checkpoints/benchmark/seed2/results.json
    uv run python docs/tools/figures.py benchmark-loss benchmark-seeds benchmark-speed

Настройки обучения (общие для всех прогонов) берутся из первого файла.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "benchmark" / "results.json"
SETUP_KEYS = ("steps", "batch_size", "block_size", "lr", "warmup_ratio", "eval_interval",
              "embed_dim", "num_layers", "num_heads", "dropout")
ABOUT = ("Результаты experiments/llm_only/benchmark.py, из них строятся рисунки docs/tools/figures.py (benchmark-*) "
         "и таблицы docs/guide/benchmark.md. Не пересчитываются при проверке рисунков: прогон занимает около 35 минут.")


def collect(paths, device):
    runs, setup = [], None
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        args = data["args"]
        setup = setup or {key: args[key] for key in SETUP_KEYS}
        for r in data["results"]:
            runs.append({
                "model": r["model"], "seed": args["seed"], "params": r["params"], "active_params": r["active_params"],
                "val_curve": [[int(step), round(loss, 4)] for step, loss in r["val_curve"]],
                "val_perplexity": round(r["val_perplexity"], 4), "train_seconds": round(r["train_seconds"], 1),
                "tokens_per_second": round(r["tokens_per_second"]),
            })
    return {"about": ABOUT, "setup": setup, "device": device, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results.json от benchmark.py, по одному на seed")
    parser.add_argument("--device", default="mps", help="устройство прогона — подпись в данных")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collect(args.results, args.device), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.out}: {args.out.stat().st_size} байт")


if __name__ == "__main__":
    main()
