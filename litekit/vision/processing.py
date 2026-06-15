"""Image transformation and processing functions."""

import logging
import io
import numpy as np
import random
from pathlib import Path
from typing import List, Literal, Tuple, Union
from PIL import Image

from .core import (
    MIN_IMAGE_DIMENSION,
    MAX_TOTAL_IMAGE_PAYLOAD_BYTES,
    _convert_to_rgb
)

logger = logging.getLogger(__name__)

def create_blank_image(
    width: int,
    height: int,
    color: Union[Tuple[int, int, int], Literal["random"]] = (255, 255, 255),
    image_mode: Literal["RGB", "RGBA", "L"] = "RGB",
) -> Image.Image:
    """Create a blank image with uniform or random color."""
    if color == "random":
        if image_mode == "L":
            fill_color = random.randint(0, 255)
        elif image_mode == "RGBA":
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        else:
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        fill_color = color
    return Image.new(image_mode, (width, height), fill_color)

def create_random_image(width: int, height: int, image_mode: Literal["RGB", "RGBA", "L"] = "RGB") -> Image.Image:
    """Create image with random pixel values (noise)."""
    if image_mode == "L":
        data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    elif image_mode == "RGBA":
        data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    else:
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, mode=image_mode)

def create_gradient_image(
    width: int,
    height: int,
    start_color: Tuple[int, int, int],
    end_color: Tuple[int, int, int],
    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal",
) -> Image.Image:
    """Create image with color gradient."""
    data = np.zeros((height, width, 3), dtype=np.uint8)
    if direction == "horizontal":
        gradient = np.linspace(0, 1, width)
        for x in range(width):
            t = gradient[x]
            data[:, x] = [int(start_color[i] * (1 - t) + end_color[i] * t) for i in range(3)]
    elif direction == "vertical":
        gradient = np.linspace(0, 1, height)
        for y in range(height):
            t = gradient[y]
            data[y, :] = [int(start_color[i] * (1 - t) + end_color[i] * t) for i in range(3)]
    else:  # diagonal
        max_dist = np.sqrt(height**2 + width**2)
        for y in range(height):
            for x in range(width):
                t = np.sqrt(x**2 + y**2) / max_dist
                data[y, x] = [int(start_color[i] * (1 - t) + end_color[i] * t) for i in range(3)]
    return Image.fromarray(data, mode="RGB")

def _estimate_base64_size(data: bytes) -> int:
    """Estimate the base64 encoded size of binary data."""
    BASE64_OVERHEAD = 50
    return int(len(data) * 4 / 3) + BASE64_OVERHEAD

DEFAULT_JPEG_QUALITY = 85
MIN_JPEG_QUALITY = 10
EXIF_ORIENTATION_TAG = 274


def _resize_image_to_quality(image_path: str, quality: int = DEFAULT_JPEG_QUALITY, scale_factor: float = 1.0) -> bytes:
    """Resize and compress an image to specified quality and dimensions."""
    with Image.open(image_path) as img:
        if scale_factor < 1.0:
            new_width = max(MIN_IMAGE_DIMENSION, int(img.width * scale_factor))
            new_height = max(MIN_IMAGE_DIMENSION, int(img.height * scale_factor))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img = _convert_to_rgb(img)
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue()

def resize_images_to_fit(image_paths: List[str]) -> List[str]:
    """Automatically resize images so total payload doesn't exceed limit."""
    total_size = 0
    for path in image_paths:
        with open(path, "rb") as f:
            total_size += _estimate_base64_size(f.read())

    if total_size <= MAX_TOTAL_IMAGE_PAYLOAD_BYTES:
        return image_paths

    quality = DEFAULT_JPEG_QUALITY
    scale_factor = 1.0
    max_iterations = 50
    for _ in range(max_iterations):
        resized_data = []
        total_resized_size = 0
        for path in image_paths:
            compressed = _resize_image_to_quality(path, quality, scale_factor)
            resized_data.append(compressed)
            total_resized_size += _estimate_base64_size(compressed)

        if total_resized_size <= MAX_TOTAL_IMAGE_PAYLOAD_BYTES:
            temp_paths = []
            for i, data in enumerate(resized_data):
                temp_path = Path(image_paths[i]).parent / f".{Path(image_paths[i]).stem}_resized_{quality}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(data)
                temp_paths.append(str(temp_path))
            return temp_paths

        if quality > MIN_JPEG_QUALITY:
            quality -= 5
        elif scale_factor > 0.1:
            scale_factor -= 0.1
        else:
            break
    raise ValueError("Cannot resize images to fit within limit.")

def square_image(image_path: str, max_size: int, background_color: Tuple[int, int, int], position: Literal["top-left", "center"] = "center") -> Image.Image:
    """Create a square image with specified background color and positioned image."""
    with Image.open(image_path) as img:
        img = _convert_to_rgb(img)
        img_width, img_height = img.size
        if img_width > max_size or img_height > max_size:
            scale = min(max_size / img_width, max_size / img_height)
            img = img.resize((int(img_width * scale), int(img_height * scale)), Image.Resampling.LANCZOS)
            img_width, img_height = img.size

        canvas = Image.new("RGB", (max_size, max_size), background_color)
        x = (max_size - img_width) // 2 if position == "center" else 0
        y = (max_size - img_height) // 2 if position == "center" else 0
        canvas.paste(img, (x, y))
        return canvas

def resize_to_dimensions(image_path: str, width: int, height: int, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Resize image to exact dimensions while maintaining aspect ratio with padding."""
    with Image.open(image_path) as img:
        img = _convert_to_rgb(img)
        img_width, img_height = img.size
        scale = min(width / img_width, height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (width, height), background_color)
        canvas.paste(img, ((width - new_width) // 2, (height - new_height) // 2))
        return canvas

def convert_format(image_path: str, target_format: Literal["JPEG", "PNG", "WEBP"], quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Convert image to a specific format with quality control."""
    with Image.open(image_path) as img:
        if target_format == "JPEG":
            img = _convert_to_rgb(img)
        output = io.BytesIO()
        img.save(output, format=target_format, quality=quality)
        return output.getvalue()

def auto_orient(image_path: str) -> Image.Image:
    """Auto-rotate image based on EXIF orientation data."""
    with Image.open(image_path) as img:
        try:
            exif_data = img.getexif()
            orientation = exif_data.get(EXIF_ORIENTATION_TAG)
        except Exception: orientation = None

        if orientation == 2: img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3: img = img.rotate(180, expand=False)
        elif orientation == 4: img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.transpose(Image.Transpose.TRANSPOSE)
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 6: img = img.rotate(270, expand=False)
        elif orientation == 7:
            img = img.transpose(Image.Transpose.TRANSVERSE)
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 8: img = img.rotate(90, expand=False)

        return _convert_to_rgb(img)

def remove_exif(image_path: str) -> Image.Image:
    """Remove all EXIF and metadata from an image."""
    with Image.open(image_path) as img:
        img = _convert_to_rgb(img)
        return Image.frombytes(img.mode, img.size, img.tobytes())

def resize_to_max_size(
    image_path: str,
    max_size: float,
    size_unit: Literal["MB", "GB"] = "MB",
    target_format: Literal["JPEG", "PNG"] = "JPEG",
    min_quality: int = 10,
) -> Image.Image:
    """Resize image to fit within a maximum file size in MB or GB."""
    max_bytes = max_size * (1024 * 1024) if size_unit == "MB" else max_size * (1024 * 1024 * 1024)
    with Image.open(image_path) as img:
        img = _convert_to_rgb(img)
        if Path(image_path).stat().st_size <= max_bytes:
            return img

        quality = DEFAULT_JPEG_QUALITY
        scale_factor = 1.0
        while quality >= min_quality:
            output = io.BytesIO()
            img.save(output, format=target_format, quality=quality, optimize=True)
            if output.tell() <= max_bytes:
                compressed = Image.open(io.BytesIO(output.getvalue()))
                compressed.load()
                return compressed
            quality -= 5

        while scale_factor > 0.1:
            scale_factor -= 0.1
            new_width = max(MIN_IMAGE_DIMENSION, int(img.width * scale_factor))
            new_height = max(MIN_IMAGE_DIMENSION, int(img.height * scale_factor))
            scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            scaled_img.save(output, format=target_format, quality=min_quality, optimize=True)
            if output.tell() <= max_bytes:
                compressed = Image.open(io.BytesIO(output.getvalue()))
                compressed.load()
                return compressed
        raise ValueError("Cannot compress image to fit within limit.")

def crop(image_path: str, left: int, top: int, right: int, bottom: int) -> Image.Image:
    """Crop a rectangular region from an image."""
    with Image.open(image_path) as img:
        return img.crop((left, top, right, bottom))
