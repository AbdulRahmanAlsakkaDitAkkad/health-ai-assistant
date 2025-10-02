# file: src/assistant/ocr/test_ingest.py

from assistant.ingest.file_ingestor import ingest_file
import os

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    pdf_path = os.path.join(base_dir, "sample_lab_report.pdf")
    image_path = os.path.join(base_dir, "sample_prescription.png")

    # Test with file path (PDF)
    print("📄 Ingesting PDF from path")
    ingest_file(pdf_path)

    print("=" * 50)

    # Test with file path (Image)
    print("🖼️ Ingesting Image from path")
    ingest_file(image_path)

    print("=" * 50)

    # Test with raw bytes (PDF)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    print("📄 Ingesting PDF from bytes")
    ingest_file(pdf_bytes, file_type="pdf")

    print("=" * 50)

    # Test with raw bytes (Image)
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    print("🖼️ Ingesting Image from bytes")
    ingest_file(img_bytes, file_type="image")
