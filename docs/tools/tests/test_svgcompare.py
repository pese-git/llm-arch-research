import base64
import io
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgcompare import EXACT, TRAINED, compare  # noqa: E402


def png(pixels):
    buf = io.BytesIO()
    mpimg.imsave(buf, np.array(pixels, dtype=float), format="png", cmap="gray", vmin=0, vmax=1)
    return base64.b64encode(buf.getvalue()).decode()


def svg(body, clip="p1", image=None, text="0.84", x="10.000"):
    picture = f'<image xlink:href="data:image/png;base64,{image}" id="img{clip}"/>' if image else ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
        f'<defs><clipPath id="{clip}"><rect x="0" y="0" width="5" height="5"/></clipPath></defs>'
        f'<g clip-path="url(#{clip})"><path d="M {x} 20 L 30 40"/><text>{text}</text>{body}{picture}</g></svg>'
    )


def test_random_ids_are_ignored():
    assert compare(svg("", clip="pa1"), svg("", clip="zz9")) == []


def test_coordinates_within_tolerance_match_and_beyond_do_not():
    assert compare(svg("", x="10.000"), svg("", x="10.005")) == []
    assert compare(svg("", x="10.000"), svg("", x="10.500")) != []
    assert compare(svg("", x="10.000"), svg("", x="10.300"), TRAINED) == []


def test_changed_number_in_label_is_found():
    assert compare(svg("", text="0.84"), svg("", text="0.91")) != []
    assert compare(svg("", text="0.84"), svg("", text="0.85"), TRAINED) == []  # в пределах 2 %


def test_changed_text_is_found_even_with_loose_tolerance():
    assert compare(svg("", text="голова 0"), svg("", text="голова 1"), TRAINED) != []
    assert compare(svg("", text="токен"), svg("", text="другой"), TRAINED) != []


def test_extra_element_is_found():
    assert compare(svg(""), svg('<path d="M 0 0"/>'), TRAINED) != []


def test_raster_is_compared_by_pixels():
    a, b = png([[0.0, 1.0]]), png([[0.0, 1.0]])
    assert compare(svg("", image=a), svg("", clip="p2", image=b)) == []
    assert compare(svg("", image=a), svg("", image=png([[0.0, 0.5]]))) != []
    assert compare(svg("", image=a), svg("", image=png([[0.0, 0.99]])), TRAINED) == []
