from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

# ✅ Load Donut model + processor once at import time
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# ✅ Questions to extract from lab reports
QUESTIONS = [
    "What is the Patient Name?",
    "What is the Date of Birth?",
    "What is the Patient ID?",
    "Who is the Doctor?",
    "What is the Date?",
    "What is the Test Ordered?",
    "What are the Results?",
    "What is the Impression?",
    "What are the Recommendations?"
]

def extract_answers_from_image_bytes(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes and returns structured answers for predefined questions.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    answers = {}
    for question in QUESTIONS:
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        decoder_input_ids = processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
        answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers[question] = answer

    return answers
