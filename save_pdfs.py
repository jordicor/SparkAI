# save_pdfs.py

import os
import re
import hashlib
import base64

import fitz  # PyMuPDF

from log_config import logger
from common import (
    CLOUDFLARE_FOR_IMAGES, CLOUDFLARE_BASE_URL, generate_signed_url_cloudflare,
    generate_user_hash, MAX_PDF_PAGES
)
from save_images import get_or_generate_img_token


async def save_pdf_locally(pdf_data: bytes, filename: str, current_user,
                           conversation_id: int, source: str = "user") -> tuple[str, str]:
    """Save PDF to disk. Returns (base_url, token_url)."""
    # generate_user_hash returns a 3-tuple: (prefix1, prefix2, full_hash)
    h1, h2, user_hash = generate_user_hash(current_user.username)

    conv_id_str = f"{conversation_id:07d}"
    c1, c2 = conv_id_str[:3], conv_id_str[3:]

    # Use pdf/uploads/ to avoid collision with pdf/ (conversation PDF exports)
    rel_dir = f"users/{h1}/{h2}/{user_hash}/files/{c1}/{c2}/pdf/uploads"
    abs_dir = os.path.join("data", rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    file_hash = hashlib.sha1(pdf_data).hexdigest()
    # Sanitize filename: keep only alphanumeric, hyphens, underscores; truncate
    safe_name = re.sub(r'[^\w\-]', '_', os.path.splitext(filename)[0])[:50]
    dest_filename = f"{file_hash}_{safe_name}.pdf"
    dest_path = os.path.join(abs_dir, dest_filename)

    if not os.path.exists(dest_path):  # Dedup by content hash
        with open(dest_path, 'wb') as f:
            f.write(pdf_data)

    rel_url = f"{rel_dir}/{dest_filename}"
    base_url = f"{CLOUDFLARE_BASE_URL or ''}{rel_url}"

    # Generate authenticated URL (same mechanism as images)
    if CLOUDFLARE_FOR_IMAGES:
        token_url = generate_signed_url_cloudflare(rel_url)
    else:
        # get_or_generate_img_token is async, takes only current_user
        img_token = await get_or_generate_img_token(current_user)
        token_url = f"{base_url}?token={img_token}"

    return base_url, token_url


def validate_pdf(pdf_data: bytes) -> int:
    """Validate PDF and return page count. Raises ValueError if invalid."""
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception:
        raise ValueError("Invalid or corrupted PDF file")
    try:
        page_count = len(doc)
    finally:
        doc.close()
    if page_count > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds {MAX_PDF_PAGES} page limit ({page_count} pages)")
    return page_count


def extract_pdf_text_local(pdf_data: bytes) -> str:
    """Extract text from PDF locally via PyMuPDF. Used only for O1 (text-only legacy provider)."""
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    try:
        pages_text = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        return "\n\n---\n\n".join(pages_text)
    finally:
        doc.close()


def get_or_extract_pdf_text(pdf_file_path: str) -> str:
    """Get extracted text for a PDF, using disk cache if available. O1 only."""
    cache_path = pdf_file_path.rsplit('.pdf', 1)[0] + '.extracted.txt'

    # Check cache first
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Extract locally via PyMuPDF
    with open(pdf_file_path, 'rb') as f:
        pdf_data = f.read()

    extracted_text = extract_pdf_text_local(pdf_data)

    # Cache to disk
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

    return extracted_text
