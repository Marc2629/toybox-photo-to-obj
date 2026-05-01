from __future__ import annotations

from PIL import Image, ImageChops


def crop_recenter_pad(
    image: Image.Image,
    foreground_ratio: float = 0.85,
    output_size: int = 512,
    background_alpha_threshold: int = 5,
) -> Image.Image:
    """Crop to non-transparent pixels, recenter, and pad to a square RGBA canvas."""
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda px: 255 if px > background_alpha_threshold else 0)
    bbox = mask.getbbox()

    if bbox is None:
        return Image.new("RGBA", (output_size, output_size), (255, 255, 255, 0))

    cropped = rgba.crop(bbox)
    max_side = max(cropped.size)
    target_subject_size = max(1, int(output_size * foreground_ratio))
    scale = min(target_subject_size / max_side, 1.0 if max_side > target_subject_size else target_subject_size / max_side)
    new_size = (max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale)))
    cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (output_size, output_size), (255, 255, 255, 0))
    paste_xy = ((output_size - cropped.width) // 2, (output_size - cropped.height) // 2)
    canvas.alpha_composite(cropped, paste_xy)
    return _trim_alpha_noise(canvas)


def rgba_to_triposr_rgb(image: Image.Image, background_level: int = 127) -> Image.Image:
    """Composite RGBA over neutral gray, matching the TripoSR preprocessing convention."""
    rgba = image.convert("RGBA")
    gray = Image.new("RGBA", rgba.size, (background_level, background_level, background_level, 255))
    gray.alpha_composite(rgba)
    return gray.convert("RGB")


def _trim_alpha_noise(image: Image.Image) -> Image.Image:
    """Normalize nearly transparent pixels to fully transparent."""
    rgba = image.convert("RGBA")
    transparent = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    alpha_diff = ImageChops.difference(rgba.getchannel("A"), transparent.getchannel("A"))
    if alpha_diff.getbbox() is None:
        return transparent
    alpha = rgba.getchannel("A").point(lambda px: 0 if px < 5 else px)
    rgba.putalpha(alpha)
    return rgba
