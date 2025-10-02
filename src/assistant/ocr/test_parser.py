import pathlib
from parser import extract_text

# Load sample PDF and image paths
base_dir = pathlib.Path(__file__).resolve().parent
pdf_path = base_dir / "sample_lab_report.pdf"
image_path = base_dir / "sample_prescription.png"

# Read and extract PDF text
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()
    text_pdf = extract_text(pdf_bytes, file_type="pdf")
    print("📄 PDF Text Extracted:")
    print(text_pdf)
    print("=" * 50)

# Read and extract Image text
with open(image_path, "rb") as f:
    img_bytes = f.read()
    text_img = extract_text(img_bytes, file_type="image")
    print("🖼️ Image Text Extracted:")
    print(text_img)
