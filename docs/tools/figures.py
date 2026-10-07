"""Строит иллюстрации учебника из кода библиотеки и формул глав.

Графики считаются детерминированно и сохраняются в docs/assets/figures/*.svg: фон
прозрачный, подписи нейтрально-серые, поэтому одна и та же картинка читается на
светлой и тёмной теме сайта и на GitHub. Главы ссылаются на них обычной
markdown-картинкой с alt-текстом.

Запуск из корня репозитория:

    uv run python docs/tools/figures.py            # все иллюстрации (с обучением моделей — несколько минут)
    uv run python docs/tools/figures.py masks      # одна, по имени файла без .svg
    uv run python docs/tools/figures.py --png DIR  # дополнительно PNG для просмотра

Иллюстрации с обучением (attention-heads, expert-load, loss-curves) используют тот же корпус,
токенизатор и Trainer, что ноутбуки; seed фиксирован, результат на CPU воспроизводим.
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


@figure("activations")
def activations():
    """ReLU, GELU (tanh) и SiLU и их производные на отрезке [−4, 4]."""
    import torch
    from llm.core.gelu import GELU

    x = torch.linspace(-4, 4, 401, requires_grad=True)
    funcs = (("ReLU", torch.relu, GREEN), ("GELU", GELU(), BLUE), ("SiLU", torch.nn.functional.silu, ORANGE))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for name, fn, color in funcs:
        y = fn(x)
        (dy,) = torch.autograd.grad(y.sum(), x)
        for ax, values in zip(axes, (y, dy)):
            ax.plot(x.detach(), values.detach(), color=color, label=name)
    for ax, title in zip(axes, ("f(z)", "f′(z)")):
        ax.set_xlabel("z"); ax.set_title(title); ax.grid(True); ax.set_xlim(-4, 4)
        ax.axhline(0, color=GRID, linewidth=0.8); ax.axvline(0, color=GRID, linewidth=0.8)
        ax.legend(loc="upper left")
    axes[1].annotate("GELU и SiLU: производная\nбольше 1 около z ≈ 1–2", (1.4, 1.13), xytext=(2.6, 0.75), textcoords="data",
                     color=INK, fontsize=8, ha="center", arrowprops=dict(arrowstyle="-", color=GRID))
    fig.tight_layout()
    return fig


@figure("lr-schedule")
def lr_schedule():
    """Линейный warmup с линейным спадом против косинусного спада; 1000 шагов, warmup 10 %."""
    total, warmup, peak = 1000, 100, 3e-4
    steps = np.arange(total + 1)
    linear = np.where(steps < warmup, steps / warmup, np.maximum(0, (total - steps) / (total - warmup))) * peak
    cosine = np.where(steps < warmup, steps / warmup, 0.5 * (1 + np.cos(np.pi * (steps - warmup) / (total - warmup)))) * peak
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(steps, linear, color=BLUE, label="линейный спад (Trainer)")
    ax.plot(steps, cosine, color=ORANGE, label="косинусный спад")
    ax.axvspan(0, warmup, color=MUTED, alpha=0.6)
    ax.annotate("warmup", (warmup / 2, peak * 1.02), ha="center", color=INK, fontsize=9)
    ax.annotate("линейный", (600, linear[600]), xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.annotate("косинусный", (600, cosine[600]), xytext=(6, -12), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("шаг обучения"); ax.set_ylabel("learning rate"); ax.set_ylim(0, peak * 1.12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax.grid(True, axis="y"); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


@figure("residual-std")
def residual_std():
    """Масштаб residual-потока по глубине у случайно инициализированной GPT-2: с делением residual-проекций на √2L и без."""
    import torch
    from llm.models.gpt import GPT2

    config = {"vocab_size": 300, "embed_dim": 256, "num_heads": 4, "num_layers": 24, "max_position_embeddings": 128, "dropout": 0.0}
    torch.manual_seed(0); scaled = GPT2(config).eval()
    torch.manual_seed(0); unscaled = GPT2(config).eval()
    for decoder in unscaled._decoders:
        for projection in (decoder._heads._layer, decoder._ff._layer2):
            torch.nn.init.normal_(projection.weight, std=0.02)
    torch.manual_seed(1); ids = torch.randint(0, config["vocab_size"], (1, 32))

    def stds(model):
        out = []
        with torch.no_grad():
            h = model._token_embeddings(ids) + model._position_embeddings(ids.size(1))
            out.append(h.std().item())
            for decoder in model._decoders:
                h = decoder(h, use_cache=False)[0]
                out.append(h.std().item())
        return out

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for model, color, label in ((scaled, BLUE, "std / √(2L) у W_O и W₂"), (unscaled, ORANGE, "везде std = 0.02")):
        values = stds(model)
        ax.plot(range(len(values)), values, color=color, marker=".", label=label)
        ax.annotate(f"{values[-1]:.2f}", (len(values) - 1, values[-1]), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_xlabel("после блока (0 — эмбеддинги)"); ax.set_ylabel("std residual-потока"); ax.set_xlim(0, 26.5)
    ax.grid(True, axis="y"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


@figure("receptive-field")
def receptive_field():
    """Какие входные позиции влияют на последнюю позицию после k слоёв при окне W = 16 (учебный конфиг Mistral)."""
    import torch
    from llm.models.mistral import Mistral

    config = {"vocab_size": 100, "embed_dim": 256, "num_q_heads": 4, "num_kv_heads": 2, "head_size": 64,
              "num_layers": 4, "max_position_embeddings": 512, "window_size": 16, "dropout": 0.0}
    T = 100
    torch.manual_seed(0); ids = torch.randint(0, config["vocab_size"], (1, T))

    def spans(model):
        captured, outputs = {}, []

        def keep_embeddings(module, inputs, output):
            output.retain_grad()
            captured["emb"] = output

        hooks = [model._token_embeddings.register_forward_hook(keep_embeddings)]
        hooks += [d.register_forward_hook(lambda m, i, o: outputs.append(o[0])) for d in model._decoders]
        model(ids)
        for h in hooks:
            h.remove()
        result = []
        for out in outputs:
            captured["emb"].grad = None
            out[0, -1].sum().backward(retain_graph=True)
            alive = (captured["emb"].grad[0].norm(dim=-1) > 0).nonzero().flatten()
            result.append((alive.min().item(), alive.max().item()))
        return result

    torch.manual_seed(0); with_window = Mistral(config).eval()
    torch.manual_seed(0); no_window = Mistral(dict(config, window_size=None)).eval()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    L, W = config["num_layers"], config["window_size"]
    for k, ((lo, hi), (lo0, hi0)) in enumerate(zip(spans(with_window), spans(no_window)), start=1):
        ax.barh(k, hi0 - lo0 + 1, left=lo0, height=0.62, color=MUTED, label="без окна" if k == 1 else None)
        ax.barh(k, hi - lo + 1, left=lo, height=0.62, color=BLUE, label=f"окно W = {W}" if k == 1 else None)
        ax.annotate(f"{lo}–{hi}: {hi - lo + 1} = k·W + 1 позиций", (T, k), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(1, L + 1)); ax.set_yticklabels([f"после слоя {k}" for k in range(1, L + 1)]); ax.invert_yaxis()
    ax.set_xlabel(f"входная позиция (всего {T} токенов, смотрим на позицию {T - 1})"); ax.set_xlim(0, T + 48)
    ax.set_xticks([0, 20, 40, 60, 80, 99]); ax.grid(True, axis="x"); ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _train(model_cls, config_file, extra=None, epochs=40, seed=0):
    """Обучает модель на учебном корпусе как ноутбуки: BPE, датасет с <eos>, Trainer. Возвращает модель, токенизатор, тексты и историю loss."""
    import contextlib
    import io
    import json

    import torch

    sys.path.insert(0, str(ROOT))
    from experiments.shared.configs import TRAIN_TEXTS
    from llm.datasets.text_with_special_tokens_dataset import TextWithSpecialTokensDataset
    from llm.tokenizers import BPETokenizer
    from llm.training.trainer import Trainer

    experiment = json.loads((ROOT / "experiments/llm_only/configs" / config_file).read_text())
    train_texts, val_texts = TRAIN_TEXTS[:12], TRAIN_TEXTS[12:]
    tokenizer = BPETokenizer()
    tokenizer.train(train_texts, vocab_size=experiment["bpe_vocab_size"], special_tokens=experiment["bpe_special_tokens"])
    dataset = lambda texts: TextWithSpecialTokensDataset(texts, tokenizer, block_size=64, add_eos=True)  # noqa: E731
    config = dict(experiment["model_config"], vocab_size=tokenizer.vocab_size, **(extra or {}))
    training = experiment["training"]
    torch.manual_seed(seed)
    model = model_cls(config)
    trainer = Trainer(model, dataset(train_texts), dataset(val_texts), lr=training["learning_rate"],
                      batch_size=training["batch_size"], num_epochs=epochs, warmup_ratio=training["warmup_ratio"])
    val_history, evaluate = [], trainer.evaluate
    trainer.evaluate = lambda: val_history.append(evaluate()) or val_history[-1]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        trainer.train()
    model.eval()
    return model, tokenizer, train_texts, trainer.loss_history, val_history


@figure("attention-heads")
def attention_heads():
    """Веса внимания четырёх голов последнего слоя LLaMA, обученной на учебном корпусе."""
    import math

    import torch
    from llm.models.llama import Llama

    model, tokenizer, train_texts, _, _ = _train(Llama, "llama_train.json")
    text = " ".join(train_texts[:2])
    ids = torch.tensor([tokenizer.encode(text)])
    tokens = [t.replace(" ", "␣") for t in tokenizer.tokenize(text)]
    attn = model._decoders[-1]._heads
    captured = {}
    hook = attn.register_forward_hook(lambda m, i, o: captured.update(x=i[0]))
    with torch.no_grad():
        model(ids)
        hook.remove()
        x = captured["x"]
        B, T, _ = x.shape
        q = attn._rope(attn._q(x).reshape(B, T, attn._num_heads, attn._head_size).transpose(1, 2))
        k = attn._rope(attn._k(x).reshape(B, T, attn._num_heads, attn._head_size).transpose(1, 2))
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn._head_size)
        scores = scores.masked_fill(~attn._tril_mask[:T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)[0]
    fig, axes = plt.subplots(1, attn._num_heads, figsize=(11, 3.7))
    for h, ax in enumerate(axes):
        ax.imshow(weights[h], cmap=SEQ_BLUE, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"голова {h}"); ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6); ax.set_yticklabels(tokens if h == 0 else [], fontsize=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("последний слой: строки — запросы, столбцы — ключи, тёмнее — больший вес", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


@figure("expert-load")
def expert_load():
    """Сколько токенов корпуса выбрало каждого эксперта в каждом слое Mixtral: без load-balancing loss и с ним."""
    import torch
    from llm.models.mixtral import Mixtral

    runs = (("router_aux_loss_coef = 0", None), ("router_aux_loss_coef = 0.01", {"router_aux_loss_coef": 0.01}))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.9), sharey=True)
    for ax, (title, extra) in zip(axes, runs):
        model, tokenizer, train_texts, _, _ = _train(Mixtral, "mixtral_train.json", extra=extra)
        E, k, L = model.config["num_experts"], model.config["top_k_experts"], model.config["num_layers"]
        counts = torch.zeros(L, E, dtype=torch.long)
        with torch.no_grad():
            for text in train_texts:
                model(torch.tensor([tokenizer.encode(text)]))
                for layer, decoder in enumerate(model._decoders):
                    chosen = torch.topk(decoder._ff.router_logits, k, dim=-1).indices
                    counts[layer] += torch.bincount(chosen.flatten(), minlength=E)
        ax.imshow(counts, cmap=SEQ_BLUE, vmin=0, vmax=counts.max().item(), aspect="auto", interpolation="nearest")
        for i in range(L):
            for j in range(E):
                value = int(counts[i, j])
                ax.text(j, i, value, ha="center", va="center", fontsize=8, color="white" if value > counts.max().item() * 0.6 else INK)
        ax.set_title(f"{title}: максимум {int(counts.max())} из {int(counts[0].sum())}", fontsize=9)
        ax.set_xlabel("эксперт"); ax.set_xticks(range(E)); ax.set_yticks(range(L)); ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("слой")
    fig.tight_layout()
    return fig


@figure("loss-curves")
def loss_curves():
    """Train и validation loss шести моделей на учебном корпусе, 40 эпох, одинаковые seed и токенизатор."""
    from llm.models.gemma import Gemma
    from llm.models.gpt import GPT, GPT2
    from llm.models.llama import Llama
    from llm.models.mistral import Mistral
    from llm.models.mixtral import Mixtral

    models = (("GPT-1", GPT, "gpt_train.json"), ("GPT-2", GPT2, "gpt2_train.json"), ("LLaMA", Llama, "llama_train.json"),
              ("Mistral", Mistral, "mistral_train.json"), ("Mixtral", Mixtral, "mixtral_train.json"), ("Gemma", Gemma, "gemma_train.json"))
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5), sharex=True, sharey=True)
    for ax, (name, cls, config_file) in zip(axes.flat, models):
        _, _, _, train_loss, val_loss = _train(cls, config_file)
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, color=BLUE, label="train")
        ax.plot(epochs, val_loss, color=ORANGE, label="validation")
        ax.set_title(name); ax.grid(True, axis="y")
        ax.annotate(f"{train_loss[-1]:.2f}", (len(train_loss), train_loss[-1]), xytext=(3, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
        ax.annotate(f"{val_loss[-1]:.2f}", (len(val_loss), val_loss[-1]), xytext=(3, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    for ax in axes[1]:
        ax.set_xlabel("эпоха")
    for ax in axes[:, 0]:
        ax.set_ylabel("cross-entropy")
    axes[0, 0].legend(loc="center right")
    fig.tight_layout()
    return fig


@figure("bpe-vocab")
def bpe_vocab():
    """Длина учебного корпуса в токенах в зависимости от размера словаря BPE: train и валидационные тексты."""
    sys.path.insert(0, str(ROOT))
    from experiments.shared.configs import TRAIN_TEXTS
    from llm.tokenizers import BPETokenizer

    train_texts, val_texts = TRAIN_TEXTS[:12], TRAIN_TEXTS[12:]
    sizes = [40, 60, 80, 120, 160, 240, 320, 426]
    rows = []
    for size in sizes:
        tokenizer = BPETokenizer()
        tokenizer.train(train_texts, vocab_size=size, special_tokens=["<unk>"])
        rows.append((sum(len(tokenizer.encode(t)) for t in train_texts), sum(len(tokenizer.encode(t)) for t in val_texts)))
    chars = (sum(map(len, train_texts)), sum(map(len, val_texts)))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (color, label) in enumerate(((BLUE, "train: 12 текстов, на них обучен словарь"), (ORANGE, "validation: 3 новых текста"))):
        values = [r[i] for r in rows]
        ax.plot(sizes, values, color=color, marker=".", label=label)
        ax.annotate(f"{chars[i] / values[-1]:.1f} симв./токен", (sizes[-1], values[-1]), xytext=(5, 9 if i == 0 else -9),
                    textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.set_xlabel("размер словаря (без специальных токенов)"); ax.set_ylabel("токенов в корпусе"); ax.set_xlim(0, 520)
    ax.grid(True, axis="y"); ax.legend(loc="upper right")
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
