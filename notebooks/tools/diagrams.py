"""Вставляет схемы учебника в ноутбуки.

Схемы живут в docs/textbook/*.md как блоки ```mermaid с полем accTitle. GitHub не рендерит
Mermaid внутри ipynb и вырезает HTML вроде <details>, поэтому в ноутбук попадает только
PNG-вложение и подпись со ссылкой на главу, где лежит исходник Mermaid.

Маркер в markdown-ячейке (путь относительно docs/, название — accTitle схемы):

    <!-- diagram: textbook/mistral.md | Архитектура Mistral -->

Скрипт заменяет всё от маркера до <!-- /diagram --> (или до конца ячейки, если
закрывающего маркера ещё нет) на картинку, подпись и закрывающий маркер с хэшем
исходника. При повторном запуске схема с тем же хэшем не перерисовывается.

Запуск из корня репозитория (нужны Node.js и сеть для первого запуска npx):

    uv run python notebooks/tools/diagrams.py                 # все ноутбуки
    uv run python notebooks/tools/diagrams.py notebooks/gpt.ipynb --force
    uv run python notebooks/tools/diagrams.py --check         # только проверить актуальность
"""
import argparse
import base64
import hashlib
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
NOTEBOOKS = ROOT / "notebooks"

MARKER = re.compile(r"<!-- diagram: (?P<doc>\S+) \| (?P<title>[^>]+?) -->")
END = re.compile(r"<!-- /diagram(?: sha=(?P<sha>[0-9a-f]+))? -->")
MMDC = ["npx", "-y", "-p", "@mermaid-js/mermaid-cli", "mmdc"]


def find_diagram(doc: Path, title: str) -> str:
    """Возвращает исходник блока ```mermaid с данным accTitle."""
    text = doc.read_text(encoding="utf-8")
    for match in re.finditer(r"```mermaid\n(.*?)\n```", text, re.S):
        block = match.group(1)
        if re.search(rf"^\s*accTitle:\s*{re.escape(title)}\s*$", block, re.M):
            return block
    raise SystemExit(f"{doc}: схема с accTitle «{title}» не найдена")


def render_png(source: str, scale: int = 2) -> bytes:
    """Рендерит Mermaid в PNG через mermaid-cli, затем сжимает палитру до 128 цветов."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "diagram.mmd", Path(tmp) / "diagram.png"
        src.write_text(source, encoding="utf-8")
        subprocess.run([*MMDC, "-i", str(src), "-o", str(out), "-b", "white", "-s", str(scale)],
                       check=True, capture_output=True, cwd=tmp)
        png = out.read_bytes()
    try:
        from PIL import Image
    except ImportError:
        return png
    image = Image.open(io.BytesIO(png)).convert("RGB")
    palette = image.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buffer = io.BytesIO()
    palette.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def chapter_link(doc: str, notebook: Path) -> str:
    """Относительная ссылка на главу из ноутбука."""
    rel = Path("..") / "docs" / doc
    name = (DOCS / doc).read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return f"[{name}]({rel.as_posix()})"


def build_block(doc: str, title: str, sha: str, attachment: str, notebook: Path) -> str:
    return "\n".join([
        f"<!-- diagram: {doc} | {title} -->",
        f"![{title}](attachment:{attachment})",
        "",
        f"*Схема «{title}» из главы {chapter_link(doc, notebook)}, там же её исходник в Mermaid.*",
        f"<!-- /diagram sha={sha} -->",
    ])


def process_cell(cell: dict, notebook: Path, force: bool, check: bool) -> list[str]:
    """Обновляет все маркеры в ячейке. Возвращает список изменений (для --check — устаревших схем)."""
    text = "".join(cell["source"])
    attachments = cell.get("attachments", {})
    changes, expected, pos = [], {}, 0
    while True:
        start = MARKER.search(text, pos)
        if not start:
            break
        doc, title = start.group("doc"), start.group("title").strip()
        end = END.search(text, start.end())
        tail = end.end() if end else len(text)
        source = find_diagram(DOCS / doc, title)
        sha = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        attachment = f"diagram-{sha}.png"
        expected[attachment] = source
        block = build_block(doc, title, sha, attachment, notebook)
        if text[start.start():tail] == block and attachment in attachments and not force:
            pos = tail
            continue
        changes.append(f"{notebook.name}: {title} ({doc})")
        if check:
            pos = tail
            continue
        text = text[:start.start()] + block + text[tail:]
        pos = start.start() + len(block)
    if check or not changes:
        return changes
    for name in [n for n in attachments if n.startswith("diagram-") and n not in expected]:
        del attachments[name]                      # устаревшие версии схем
    for name, source in expected.items():
        if name not in attachments or force:
            attachments[name] = {"image/png": base64.b64encode(render_png(source)).decode("ascii")}
    cell["attachments"] = attachments
    cell["source"] = text
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="ноутбуки; по умолчанию все в notebooks/")
    parser.add_argument("--force", action="store_true", help="перерисовать даже неизменившиеся схемы")
    parser.add_argument("--check", action="store_true", help="не менять файлы, вернуть 1, если есть устаревшие схемы")
    args = parser.parse_args()

    import nbformat

    notebooks = args.notebooks or sorted(NOTEBOOKS.glob("*.ipynb"))
    stale = []
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        changed = []
        for cell in nb.cells:
            if cell.cell_type == "markdown" and "<!-- diagram:" in "".join(cell.source):
                changed += process_cell(cell, path, args.force, args.check)
        if changed and not args.check:
            nbformat.write(nb, path)
        stale += changed
    for line in stale:
        print(("устарела: " if args.check else "обновлена: ") + line)
    if not stale:
        print("все схемы актуальны")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
