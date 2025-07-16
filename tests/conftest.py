import pytest
from PIL import Image


@pytest.fixture
def temp_image(tmp_path):
    """Return a temporary RGB image path."""
    path = tmp_path / "img.png"
    Image.new("RGB", (200, 100), color=(255, 0, 0)).save(path)
    return path
