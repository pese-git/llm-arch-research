"""Строит иллюстрации учебника из кода библиотеки и формул глав.

Графики считаются детерминированно и сохраняются в docs/assets/figures/*.svg: фон
прозрачный, подписи нейтрально-серые, поэтому одна и та же картинка читается на
светлой и тёмной теме сайта и на GitHub. Главы ссылаются на них обычной
markdown-картинкой с alt-текстом.

Запуск из корня репозитория:

    uv run python docs/tools/figures.py            # все иллюстрации
    uv run python docs/tools/figures.py masks      # одна, по имени файла без .svg
    uv run python docs/tools/figures.py --png DIR  # дополнительно PNG для просмотра
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "figures"

# Палитра: три категориальных цвета, проверенных на светлой и тёмной поверхности
# (dataviz: CVD ΔE и контраст), нейтральный серый для текста и сетки.
BLUE, ORANGE, GREEN = "#3987e5", "#d95926", "#199e70"
INK, GRID, MUTED = "#7a7975", "#b0afa9", "#e8e7e3"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "savefig.transparent": True, "svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.alpha": 0.35, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
})

FIGURES = {}


def figure(name):
    def register(fn):
        FIGURES[name] = fn
        return fn
    return register


def _cells(ax, mask, title):
    """Матрица маски: синие клетки разрешены, серые закрыты; строки — запросы i, столбцы — ключи j."""
    T = mask.shape[0]
    ax.imshow(mask, cmap=ListedColormap([MUTED, BLUE]), vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticks(np.arange(-0.5, T, 1), minor=True); ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5, alpha=1)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("ключ j"); ax.set_ylabel("запрос i"); ax.set_title(title)


@figure("masks")
def masks():
    """Causal-маска и маска скользящего окна (W = 4) для T = 8: что видит каждый запрос."""
    T, W = 8, 4
    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    causal = (j <= i).astype(int)
    window = ((j <= i) & (i - j <= W)).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    _cells(axes[0], causal, "causal: j ≤ i")
    _cells(axes[1], window, f"скользящее окно, W = {W}: 0 ≤ i − j ≤ W")
    fig.tight_layout()
    return fig


@figure("rope-frequencies")
def rope_frequencies():
    """Частоты RoPE: cos(t·θ_i) по позициям и парам и период каждой пары для двух баз."""
    d_h, T = 64, 128
    pairs = np.arange(d_h // 2)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})

    theta = 1e4 ** (-2 * pairs / d_h)
    cos = np.cos(np.arange(T)[:, None] * theta[None, :])
    ax = axes[0]
    im = ax.imshow(cos.T, aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xlabel("позиция t"); ax.set_ylabel("пара координат i"); ax.set_title("cos(t·θᵢ), d_h = 64, base = 10⁴")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, 0, 1])
    cbar.outline.set_visible(False); cbar.ax.tick_params(color=GRID, labelcolor=INK)

    ax = axes[1]
    for base, color, label in ((1e4, BLUE, "base = 10⁴"), (1e6, ORANGE, "base = 10⁶")):
        period = 2 * math.pi / base ** (-2 * pairs / d_h)
        ax.plot(pairs, period, color=color, label=label)
        ax.annotate(label, (pairs[-1], period[-1]), xytext=(-8, -12), textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.axhline(512, color=GRID, linestyle="--", linewidth=1)
    ax.annotate("контекст 512", (0, 512), xytext=(2, 4), textcoords="offset points", color=INK, fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("пара координат i"); ax.set_ylabel("период 2π/θᵢ, позиций")
    ax.set_title("за сколько позиций пара делает оборот"); ax.grid(True, which="major"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


@figure("sampling")
def sampling():
    """Температура меняет форму распределения; top-k и top-p отрезают хвост по-разному."""
    z = np.array([3.0, 2.2, 1.8, 1.0, 0.7, 0.4, 0.2, 0.0, -0.3, -0.6, -1.0, -1.5])
    ranks = np.arange(1, len(z) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    ax = axes[0]
    width = 0.27
    for k, (tau, color) in enumerate(((0.5, BLUE), (1.0, ORANGE), (2.0, GREEN))):
        p = np.exp(z / tau); p /= p.sum()
        ax.bar(ranks + (k - 1) * width, p, width=width * 0.92, color=color, label=f"τ = {tau}")
    ax.set_xticks(ranks); ax.set_xlabel("токены по убыванию логита"); ax.set_ylabel("вероятность")
    ax.set_title("температура: одни логиты, три распределения"); ax.grid(True, axis="y"); ax.legend()

    ax = axes[1]
    p = np.exp(z); p /= p.sum()
    cum_before = np.cumsum(p) - p
    top_p, top_k = 0.9, 5
    in_nucleus = cum_before < top_p
    ax.bar(ranks, p, color=[BLUE if keep else MUTED for keep in in_nucleus], width=0.8, label="вероятность, τ = 1")
    ax.plot(ranks, np.cumsum(p), color=ORANGE, marker=".", label="накопленная сумма")
    ax.axhline(top_p, color=ORANGE, linestyle="--", linewidth=1)
    ax.annotate(f"top_p = {top_p}: ядро из {in_nucleus.sum()} токенов", (ranks[-1], top_p), xytext=(0, 4),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.axvline(top_k + 0.5, color=GREEN, linestyle=":", linewidth=1.5)
    ax.annotate(f"top_k = {top_k}: граница", (top_k + 0.5, 0.66), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xticks(ranks); ax.set_xlabel("токены по убыванию вероятности"); ax.set_ylim(0, 1.05)
    ax.set_title("top-p оставляет ядро, top-k — k токенов"); ax.grid(True, axis="y")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    fig.tight_layout()
    return fig


@figure("kv-cache")
def kv_cache():
    """Размер KV-кэша одной последовательности во float16 для MHA, GQA и MQA."""
    models = [
        ("LLaMA 7B: MHA, 32 K/V-головы", dict(layers=32, kv_heads=32, d_h=128), BLUE, "-"),
        ("Mistral 7B: GQA, 8 K/V-голов", dict(layers=32, kv_heads=8, d_h=128), ORANGE, "-"),
        ("Gemma 2B: MQA, 1 K/V-голова", dict(layers=18, kv_heads=1, d_h=256), GREEN, "-"),
    ]
    T = np.arange(0, 32769, 512)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for label, cfg, color, style in models:
        gib = 2 * cfg["layers"] * cfg["kv_heads"] * cfg["d_h"] * T * 2 / 2**30
        ax.plot(T, gib, color=color, linestyle=style, label=label)
        dy = -10 if cfg["kv_heads"] == 1 else -3      # подпись Gemma уходит под линию, чтобы не слиться с окном Mistral
        ax.annotate(f"{gib[-1]:.2f} ГиБ", (T[-1], gib[-1]), xytext=(4, dy), textcoords="offset points", color=INK, fontsize=9)
    mistral_window = 2 * 32 * 8 * 128 * np.minimum(T, 4096) * 2 / 2**30
    ax.plot(T, mistral_window, color=ORANGE, linestyle="--", label="Mistral 7B с окном W = 4096")
    ax.annotate(f"с окном: {mistral_window[-1]:.1f} ГиБ и не растёт", (20000, mistral_window[-1]), xytext=(0, 6),
                textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xlabel("длина последовательности, токенов"); ax.set_ylabel("KV-кэш, ГиБ (float16)")
    ax.set_xlim(0, 36500); ax.set_xticks([0, 4096, 8192, 16384, 32768]); ax.grid(True, axis="y"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="какие иллюстрации строить (по умолчанию все)")
    parser.add_argument("--png", type=Path, help="папка для PNG-копий (для просмотра)")
    args = parser.parse_args()
    names = args.names or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"нет таких иллюстраций: {unknown}; есть {list(FIGURES)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig = FIGURES[name]()
        fig.savefig(OUT / f"{name}.svg", format="svg", metadata={"Date": None})
        if args.png:
            args.png.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.png / f"{name}.png", dpi=150, facecolor="white")
        plt.close(fig)
        print(f"{OUT.relative_to(ROOT) / (name + '.svg')}: {(OUT / (name + '.svg')).stat().st_size // 1024} КиБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
