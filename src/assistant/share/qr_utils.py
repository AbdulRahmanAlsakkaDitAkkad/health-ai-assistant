# src/assistant/share/qr_utils.py
from __future__ import annotations
import base64
import qrcode
from io import BytesIO

def make_qr_png_base64(data_str: str, box_size: int = 8, border: int = 2) -> str:
    """
    Create a QR PNG (bytes) from a string, then return base64-encoded PNG.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    return base64.b64encode(png_bytes).decode("utf-8")
