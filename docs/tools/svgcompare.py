"""Сравнивает два SVG по смыслу, а не побайтно.

Два прогона figures.py на одном коде дают разные байты: matplotlib выдаёт случайные id
(clip-path, image), а числа в обучаемых иллюстрациях зависят от платформы. Поэтому дерево
сверяется так:

- теги и число потомков совпадают (то же число линий, столбцов, подписей, делений);
- id и ссылки на них (`id`, `url(#…)`, `#…`) игнорируются;
- текст и атрибуты делятся на числа и остальное: остальное совпадает точно (подписи, цвета,
  названия токенов), числа — с допуском `atol` (координаты) и `rtol` (значения в подписях);
- растровые вставки (карты внимания) сравниваются по пикселям, а не по байтам PNG.
"""
import base64
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import matplotlib.image as mpimg
import numpy as np

NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
IGNORED_ATTRS = {"id"}
IMAGE_HREF = "{http://www.w3.org/1999/xlink}href"


@dataclass(frozen=True)
class Tolerance:
    """atol — координаты и размеры в пикселях SVG; rtol — числа в тексте; pixel — яркость пикселей 0…1."""

    atol: float = 0.01
    rtol: float = 1e-9
    pixel: float = 1e-6


EXACT = Tolerance()
TRAINED = Tolerance(atol=1e9, rtol=1.0, pixel=1.0)  # смотрим только на структуру и нечисловой текст


def _strip_refs(value: str) -> str:
    value = re.sub(r"url\(#[^)]*\)", "url(#)", value)
    return "#" if re.fullmatch(r"#\S+", value.strip()) else value


def _numbers_match(a: str, b: str, tol: Tolerance, relative: bool) -> bool:
    if NUMBER.sub("#", a) != NUMBER.sub("#", b):
        return False
    xs, ys = (list(map(float, NUMBER.findall(s))) for s in (a, b))
    return all(
        abs(x - y) <= (tol.rtol * max(abs(x), abs(y)) if relative else tol.atol)
        for x, y in zip(xs, ys)
    )


def _pixels(href: str) -> np.ndarray:
    return mpimg.imread(io.BytesIO(base64.b64decode(href.split(",", 1)[1].replace("\n", ""))), format="png")


def _walk(a: ET.Element, b: ET.Element, tol: Tolerance, path: str, problems: list):
    name = a.tag.split("}")[-1]
    path = f"{path}/{name}" + (f"#{a.get('id')}" if a.get("id") and name == "g" else "")
    if a.tag != b.tag:
        problems.append(f"{path}: тег {a.tag!r} ≠ {b.tag!r}")
        return
    if set(a.attrib) != set(b.attrib):
        problems.append(f"{path}: атрибуты {sorted(a.attrib)} ≠ {sorted(b.attrib)}")
        return
    for key in a.attrib:
        if key in IGNORED_ATTRS:
            continue
        if key == IMAGE_HREF and a.get(key).startswith("data:"):
            pa, pb = _pixels(a.get(key)), _pixels(b.get(key))
            if pa.shape != pb.shape or np.abs(pa - pb).max() > tol.pixel:
                problems.append(f"{path}: растр отличается")
            continue
        va, vb = _strip_refs(a.get(key)), _strip_refs(b.get(key))
        if not _numbers_match(va, vb, tol, relative=False):
            problems.append(f"{path}[{key}]: {va[:60]!r} ≠ {vb[:60]!r}")
    ta, tb = (a.text or "").strip(), (b.text or "").strip()
    if not _numbers_match(ta, tb, tol, relative=True):
        problems.append(f"{path}: текст {ta[:60]!r} ≠ {tb[:60]!r}")
    if len(a) != len(b):
        problems.append(f"{path}: потомков {len(a)} ≠ {len(b)}")
        return
    for ca, cb in zip(a, b):
        _walk(ca, cb, tol, path, problems)


def compare(expected: str, actual: str, tol: Tolerance = EXACT) -> list:
    """Список расхождений между двумя SVG (пустой — совпадают по смыслу)."""
    problems: list = []
    _walk(ET.fromstring(expected), ET.fromstring(actual), tol, "", problems)
    return problems
