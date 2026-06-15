"""Compatibility layer for image utilities.

This module is deprecated. Use the 'litekit.vision' package instead.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .vision import (
    IMAGE_MIME_TYPE,
    is_valid_image,
    is_valid_size,
    is_valid_dimensions,
    encode_to_base64,
    b64_to_pil,
    pil_to_b64,
    cv2_to_pil,
    pil_to_cv2,
    get_image_info,
    save_image,
    save_images_batch,
    create_blank_image,
    create_random_image,
    create_gradient_image,
    resize_images_to_fit,
    square_image,
    resize_to_dimensions,
    convert_format,
    crop,
    auto_orient,
    remove_exif,
    resize_to_max_size,
    collect_images,
    collect_images_with_info,
)

class ImageUtils:
    """Compatibility class for vision utilities."""

    IMAGE_MIME_TYPE: str = IMAGE_MIME_TYPE

    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        return encode_to_base64(image_path)

    @staticmethod
    def is_valid_image(path: str) -> bool:
        return is_valid_image(path)

    @staticmethod
    def is_valid_size(path: str) -> bool:
        return is_valid_size(path)

    @staticmethod
    def is_valid_dimensions(path: str) -> bool:
        return is_valid_dimensions(path)

    @staticmethod
    def create_blank_image(*args: Any, **kwargs: Any) -> Image.Image:
        return create_blank_image(*args, **kwargs)

    @staticmethod
    def create_random_image(*args: Any, **kwargs: Any) -> Image.Image:
        return create_random_image(*args, **kwargs)

    @staticmethod
    def create_gradient_image(*args: Any, **kwargs: Any) -> Image.Image:
        return create_gradient_image(*args, **kwargs)

    @staticmethod
    def resize_images_to_fit(image_paths: List[str]) -> List[str]:
        return resize_images_to_fit(image_paths)

    @staticmethod
    def square_image(*args: Any, **kwargs: Any) -> Image.Image:
        return square_image(*args, **kwargs)

    @staticmethod
    def resize_to_dimensions(*args: Any, **kwargs: Any) -> Image.Image:
        return resize_to_dimensions(*args, **kwargs)

    @staticmethod
    def convert_format(*args: Any, **kwargs: Any) -> Image.Image:
        return convert_format(*args, **kwargs)

    @staticmethod
    def crop(*args: Any, **kwargs: Any) -> Image.Image:
        return crop(*args, **kwargs)

    @staticmethod
    def b64_to_pil(b64_string: str) -> Image.Image:
        return b64_to_pil(b64_string)

    @staticmethod
    def pil_to_b64(*args: Any, **kwargs: Any) -> str:
        return pil_to_b64(*args, **kwargs)

    @staticmethod
    def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
        return cv2_to_pil(cv_image)

    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        return pil_to_cv2(image)

    @staticmethod
    def get_image_info(image_path: str) -> Dict:
        return get_image_info(image_path)

    @staticmethod
    def save_image(*args: Any, **kwargs: Any) -> str:
        return save_image(*args, **kwargs)

    @staticmethod
    def save_images_batch(*args: Any, **kwargs: Any) -> List[str]:
        return save_images_batch(*args, **kwargs)

    @staticmethod
    def auto_orient(image_path: str) -> str:
        return auto_orient(image_path)

    @staticmethod
    def remove_exif(image_path: str) -> str:
        return remove_exif(image_path)

    @staticmethod
    def resize_to_max_size(*args: Any, **kwargs: Any) -> Image.Image:
        return resize_to_max_size(*args, **kwargs)

    @staticmethod
    def collect_images(*args: Any, **kwargs: Any) -> List[str]:
        return collect_images(*args, **kwargs)

    @staticmethod
    def collect_images_with_info(*args: Any, **kwargs: Any) -> List[Dict]:
        return collect_images_with_info(*args, **kwargs)
