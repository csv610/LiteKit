"""Vision utilities package for image processing and analysis."""

from .core import (
    IMAGE_MIME_TYPE,
    MAX_IMAGE_SIZE_MB,
    MAX_IMAGE_SIZE_BYTES,
    MAX_TOTAL_IMAGE_PAYLOAD_MB,
    MAX_TOTAL_IMAGE_PAYLOAD_BYTES,
    MIN_IMAGE_DIMENSION,
)
from .validation import (
    is_valid_image,
    is_valid_size,
    is_valid_dimensions,
)
from .io import (
    encode_to_base64,
    b64_to_pil,
    pil_to_b64,
    cv2_to_pil,
    pil_to_cv2,
    get_image_info,
    save_image,
    save_images_batch,
)
from .processing import (
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
)
from .collection import (
    collect_images,
    collect_images_with_info,
)

__all__ = [
    "IMAGE_MIME_TYPE",
    "MAX_IMAGE_SIZE_MB",
    "MAX_IMAGE_SIZE_BYTES",
    "MAX_TOTAL_IMAGE_PAYLOAD_MB",
    "MAX_TOTAL_IMAGE_PAYLOAD_BYTES",
    "MIN_IMAGE_DIMENSION",
    "is_valid_image",
    "is_valid_size",
    "is_valid_dimensions",
    "encode_to_base64",
    "b64_to_pil",
    "pil_to_b64",
    "cv2_to_pil",
    "pil_to_cv2",
    "get_image_info",
    "save_image",
    "save_images_batch",
    "create_blank_image",
    "create_random_image",
    "create_gradient_image",
    "resize_images_to_fit",
    "square_image",
    "resize_to_dimensions",
    "convert_format",
    "crop",
    "auto_orient",
    "remove_exif",
    "resize_to_max_size",
    "collect_images",
    "collect_images_with_info",
]
