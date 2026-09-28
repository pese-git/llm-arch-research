"""
Source-level checks for the llm package.
"""

import pathlib
import warnings

import pytest

import llm

SOURCE_FILES = sorted(pathlib.Path(llm.__file__).parent.rglob("*.py"))


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=[str(p.relative_to(p.parents[1])) for p in SOURCE_FILES]
)
def test_no_invalid_escape_sequences(path):
    """
    Строки с LaTeX-формулами должны быть сырыми (r\"\"\"...\"\"\"): иначе
    "\\beta" превращается в backspace + "eta", а "\\eta" дает SyntaxWarning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
