from __future__ import annotations

from io import BytesIO
from functools import lru_cache

from PIL import Image
from rembg import new_session, remove


@lru_cache(maxsize=1)
def _session():
    return new_session()


def remove_background(image: Image.Image) -> Image.Image:
    """Remove the image background locally and return an RGBA image."""
    result = remove(image.convert("RGBA"), session=_session())
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    if isinstance(result, bytes):
        return Image.open(BytesIO(result)).convert("RGBA")
    return Image.open(result).convert("RGBA")
