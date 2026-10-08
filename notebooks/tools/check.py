"""Проверяет сохранённые ноутбуки, не запуская их.

Ноутбуки лежат в репозитории с выводами, чтобы читать их на GitHub. Скрипт ловит то, что
портит такое чтение:

  * файл не проходит nbformat.validate;
  * в выводах есть ошибка (output_type == "error") или трассировка в тексте;
  * ячейка с кодом не выполнялась или счётчики идут не подряд 1, 2, 3, … — значит, ноутбук
    не перезапускали целиком («Restart Kernel and Run All Cells»);
  * в выводах остались полосы tqdm — шум, который не читается на GitHub;
  * файл больше лимита: в выводах застряли лишние картинки.

Запуск из корня репозитория (нужен только nbformat):

    uv run python notebooks/tools/check.py                    # все ноутбуки
    uv run python notebooks/tools/check.py notebooks/gpt.ipynb

Код выхода 1, если что-то найдено. Актуальность схем проверяет diagrams.py --check,
выполнимость ячеек — jupyter nbconvert --execute (см. notebooks/README.md).
"""
import argparse
import re
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1]
MAX_BYTES = 600 * 1024                 # самый большой ноутбук сейчас ~350 КиБ
TQDM = re.compile(r"\d+%\|[^|]*\||\d+(?:\.\d+)?(?:it/s|s/it)\]")


def output_text(output) -> str:
    """Весь текст вывода ячейки: потоки, результаты и text/plain из display_data."""
    if output.output_type == "stream":
        return "".join(output.text) if not isinstance(output.text, str) else output.text
    data = output.get("data", {})
    text = data.get("text/plain", "")
    return "".join(text) if not isinstance(text, str) else text


def check_notebook(path: Path) -> list[str]:
    import nbformat

    problems = []
    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"файл {size // 1024} КиБ, лимит {MAX_BYTES // 1024} КиБ — лишние картинки в выводах?")
    nb = nbformat.read(path, as_version=4)
    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as err:
        problems.append(f"не проходит nbformat.validate: {str(err).splitlines()[0]}")

    expected = 1
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or not "".join(cell.source).strip():
            continue
        where = f"ячейка {index}"
        if cell.execution_count != expected:
            problems.append(f"{where}: execution_count={cell.execution_count}, ожидалось {expected} — "
                            "перезапустите ноутбук целиком")
        expected += 1
        for output in cell.outputs:
            if output.output_type == "error":
                problems.append(f"{where}: ошибка в выводе ({output.ename}: {output.evalue})")
                continue
            text = output_text(output)
            if "Traceback (most recent call last)" in text:
                problems.append(f"{where}: трассировка в выводе")
            if TQDM.search(text):
                problems.append(f"{where}: полоса tqdm в выводе — перенаправьте stdout/stderr")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="ноутбуки; по умолчанию все в notebooks/")
    args = parser.parse_args()

    paths = args.notebooks or sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        print("ноутбуки не найдены", file=sys.stderr)
        return 1
    failed = False
    for path in paths:
        problems = check_notebook(path)
        print(f"{'FAIL' if problems else 'ok  '} {path.name}")
        for problem in problems:
            print(f"     - {problem}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
