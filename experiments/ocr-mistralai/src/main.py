"""Typed and refactored version of the Mistral OCR utility."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List
from datetime import datetime

import fire
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import OCRImageObject, OCRPageObject, OCRResponse

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _init_client() -> Mistral:
    """Instantiate a Mistral client using the API key from environment."""
    api_key: str | None = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable MISTRAL_API_KEY is missing")
    return Mistral(api_key=api_key)


def _upload_pdf(client: Mistral, pdf_path: Path) -> str:
    """Upload a PDF and return its server‑side file ID."""
    response = client.files.upload(
        file={
            "file_name": pdf_path.name,
            "content": pdf_path.read_bytes(),
            "content_type": "application/pdf",
        },
        purpose="ocr",
    )
    return response.id


def _get_signed_url(client: Mistral, file_id: str) -> str:
    """Retrieve a temporary, publicly accessible URL for a file."""
    return client.files.get_signed_url(file_id=file_id).url


def _run_ocr(client: Mistral, document_url: str) -> OCRResponse:
    """Invoke the OCR endpoint and return the raw response."""
    return client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": document_url},
        include_image_base64=True,  # essential for image extraction
    )


def _save_image(page_image: OCRImageObject, dest_dir: Path) -> str:
    """
    Decode a base64‑encoded image and write it to disk.

    Returns the relative path used inside Markdown.
    """
    header, encoded = page_image.image_base64.split(",", 1)
    suffix = header.split(";")[0].split("/")[-1]  # e.g. 'png' or 'jpeg'
    file_name = f"{page_image.id}.{suffix}"
    (dest_dir / file_name).write_bytes(base64.b64decode(encoded))
    return f"images/{file_name}"


def _page_markdown_with_images(page: OCRPageObject, image_dir: Path) -> str:
    """
    Replace image placeholders in a page's Markdown with relative paths.

    Mistral returns placeholders like ![](img_id). We map each ID to the
    corresponding PNG/JPEG file written on disk.
    """
    md: str = page.markdown
    for img in page.images:
        rel_path: str = _save_image(img, image_dir)
        md = md.replace(f"![]({img.id})", f"![]({rel_path})").replace(
            f"![{img.id}]({img.id})", f"![]({rel_path})"
        )
    return md


def _write_markdown(ocr: OCRResponse, output_dir: Path) -> Path:
    """Serialize the OCR result into a single Markdown document."""
    output_dir.mkdir(exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    pages_md: List[str] = [
        _page_markdown_with_images(page, image_dir) for page in ocr.pages
    ]
    md_path = output_dir / "document.md"
    md_path.write_text("\n\n".join(pages_md), encoding="utf-8")
    return md_path


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(file_path: str) -> None:
    """
    Perform OCR on a PDF and export Markdown plus extracted images.

    Args:
        file_path: Path to the target PDF.
    """
    load_dotenv()
    pdf_file = Path(file_path).expanduser().resolve()
    if not pdf_file.exists():
        raise FileNotFoundError(pdf_file)

    with _init_client() as client:
        file_id: str = _upload_pdf(client, pdf_file)
        signed_url: str = _get_signed_url(client, file_id)
        ocr: OCRResponse = _run_ocr(client, signed_url)

    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path("data") / today
    md_file: Path = _write_markdown(ocr, output_dir)

    print(f"OCR complete → {md_file}")
    print(f"Usage info: {ocr.usage_info}")

if __name__ == "__main__":
    fire.Fire(main)
