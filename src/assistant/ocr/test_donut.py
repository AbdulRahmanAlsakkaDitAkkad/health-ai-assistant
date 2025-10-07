from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os

# Step 1: Load model + processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Step 2: Define your image path
image_path = "src/assistant/ocr/fake_lab_report.JPG"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ File not found: {image_path}")

# Step 3: Load and convert image
image = Image.open(image_path).convert("RGB")

# Step 4: Define your questions
questions = [
    "What is the Patient Name?",
    "What is the Date of Birth?",
    "What is the Patient ID?",
    "Who is the Doctor?",
    "What is the Date?",
    "What is the Test Ordered?",
    "What is the Hemoglobin result?",
    "What is the WBC Count?",
    "What is the Platelets count?",
    "What is the Hematocrit result?",
    "What is the Impression?",
    "What are the Recommendations?"
]

# Step 5: Process each question
print(f"\n🧾 Extracting answers from: {image_path}\n")

for question in questions:
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    pixel_values = processor(image, return_tensors="pt").pixel_values
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # 🔁 Increase max_length to handle long answers
    outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=1024)

    result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔧 Clean result (remove echoed question if present)
    cleaned = result.replace(question.lower(), "").strip()

    # Step 6: Print answer
    print(f"{question}\n→ {cleaned}\n{'-'*60}")
