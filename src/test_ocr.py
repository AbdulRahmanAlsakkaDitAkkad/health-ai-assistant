from assistant.ocr.parser import extract_text
from pathlib import Path

def test_pdf():
    file_path = Path("sample.pdf")  # <- Use a sample test file
    file_bytes = file_path.read_bytes()
    text = extract_text(file_bytes, file_type="pdf")
    print("Extracted PDF text:\n", text)

def test_image():
    file_path = Path("sample_image.png")  # <- Use a sample test file
    file_bytes = file_path.read_bytes()
    text = extract_text(file_bytes, file_type="image")
    print("Extracted image text:\n", text)

# Choose one or both
test_pdf()
test_image()
