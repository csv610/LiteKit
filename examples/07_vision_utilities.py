"""Example: Image utilities — encoding, resizing, format conversion."""

from litekit import ImageUtils
from litekit.vision import (
    collect_images,
    resize_to_dimensions,
    convert_format,
    encode_to_base64,
    save_image,
)

import tempfile, os
from PIL import Image

tmp = tempfile.mkdtemp()

img = Image.new("RGB", (200, 150), color=(255, 0, 0))
sample_path = os.path.join(tmp, "sample.jpg")
img.save(sample_path, "JPEG")

b64 = encode_to_base64(sample_path)
print(f"Base64 length: {len(b64)} chars")

resized = resize_to_dimensions(sample_path, width=100, height=100)
save_image(resized, os.path.join(tmp, "resized.png"))
print(f"Resized image saved to {tmp}/resized.png")

webp_bytes = convert_format(sample_path, target_format="WEBP", quality=80)
print(f"WebP size: {len(webp_bytes)} bytes")

info = ImageUtils.get_image_info(sample_path)
print(info)

paths = collect_images(tmp, recursive=False, sort_by="name")
print(f"Found {len(paths)} image(s) in temp dir")
