from pathlib import Path

from PIL import Image
import sys
import types
import importlib.util

tag_mod = types.ModuleType("imgutils.tagging")
tag_mod.get_wd14_tags = lambda *a, **k: (None, [], [])
imgutils_mod = types.ModuleType("imgutils")
imgutils_mod.tagging = tag_mod
sys.modules.setdefault("imgutils", imgutils_mod)
sys.modules.setdefault("imgutils.tagging", tag_mod)

module_path = Path(__file__).resolve().parents[1] / "run_flux_lora.py"
spec = importlib.util.spec_from_file_location("run_flux_lora", module_path)
rfl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rfl)


def test_resize_if_needed_no_resize(temp_image, tmp_path):
    dst = tmp_path / "out.png"
    rfl.resize_if_needed(temp_image, dst, short=100)
    with Image.open(dst) as im:
        assert im.size == (200, 100)


def test_resize_if_needed_scale_down(temp_image, tmp_path):
    dst = tmp_path / "out_small.png"
    rfl.resize_if_needed(temp_image, dst, short=50)
    with Image.open(dst) as im:
        assert im.size == (100, 50)


def test_tag_with_wd14(monkeypatch, temp_image):
    def fake_get_wd14_tags(*args, **kwargs):
        return None, ["cute"], ["1girl"]

    monkeypatch.setattr(rfl, "get_wd14_tags", fake_get_wd14_tags)
    rfl.tag_with_wd14([Path(temp_image)], trigger="foo", min_prob=0.5)

    caption = Path(temp_image).with_suffix(".txt").read_text()
    assert caption == "foo, 1girl, cute"
