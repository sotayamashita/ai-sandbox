import base64
import io
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

MAX_PIXEL = 1568 * 1568


def convert_pdf_to_images(pdf_path: str, output_dir: Path) -> list[Image.Image]:
    images = convert_from_path(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        out_path = output_dir / f"page_{i + 1}.png"
        img.save(out_path, "PNG")
    return images


def resize_image(image: Image.Image) -> None:
    w, h = image.size
    if w * h > MAX_PIXEL:
        scale = (MAX_PIXEL / (w * h)) ** 0.5
        image.thumbnail((int(w * scale), int(h * scale)))


def convert_image_to_base64(image_path: str) -> str:
    image = Image.open(image_path)
    resize_image(image)
    with io.BytesIO() as buffer:
        fmt = image.format or "PNG"
        image.save(buffer, format=fmt)
        return (
            f"data:image/{fmt.lower()};base64,"
            + base64.b64encode(buffer.getvalue()).decode()
        )
