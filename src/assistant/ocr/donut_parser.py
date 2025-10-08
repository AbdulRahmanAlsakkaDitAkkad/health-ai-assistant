# from transformers import DonutProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import io

# # ✅ Load Donut model + processor once at import time
# processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
# model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# # ✅ Questions to extract from lab reports
# QUESTIONS = [
#     "What is the Patient Name?",
#     "What is the Date of Birth?",
#     "What is the Patient ID?",
#     "Who is the Doctor?",
#     "What is the Date?",
#     "What is the Test Ordered?",
#     "What are the Results?",
#     "What is the Impression?",
#     "What are the Recommendations?"
# ]

# def extract_answers_from_image_bytes(image_bytes: bytes) -> dict:
#     """
#     Takes raw image bytes and returns structured answers for predefined questions.
#     """
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     pixel_values = processor(image, return_tensors="pt").pixel_values

#     answers = {}
#     for question in QUESTIONS:
#         prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
#         decoder_input_ids = processor.tokenizer(
#             prompt, add_special_tokens=False, return_tensors="pt"
#         ).input_ids
#         outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
#         answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         answers[question] = answer

#     return answers













# file: src/assistant/ocr/donut_parser.py

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import torch # Explicitly import torch
from typing import List, Dict
import warnings

# --- Global Model Setup (Load once) ---
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"

try:
    # Load Donut model + processor once at import time
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # CRITICAL: Set device for stability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Set model to evaluation mode
    
    print(f"✅ Donut Model loaded and set to {device.upper()}: {MODEL_NAME}")
    DONUT_AVAILABLE = True
except Exception as e:
    warnings.warn(f"Donut model failed to load. Error: {e}")
    DONUT_AVAILABLE = False


def extract_answers_from_image_bytes(image_bytes: bytes, questions: List[str]) -> Dict[str, str]:
    """
    Takes raw image bytes and returns structured answers for a dynamic list of questions.
    """
    if not DONUT_AVAILABLE:
        # Returns a message if the model failed to load during startup
        return {"Donut_Status": "Model not loaded. Structured extraction unavailable."} 

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Encode image and ensure it's on the correct device
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)

    answers = {}
    
    for question in questions:
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        
        # 2. Tokenize prompt and ensure it's on the correct device
        decoder_input_ids = processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(model.device)

        # 3. Generate the answer
        with torch.no_grad(): # Good practice for inference
            outputs = model.generate(
                pixel_values, 
                decoder_input_ids=decoder_input_ids, 
                max_length=512
            )
        
        # 4. Decode and clean
        answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_answer = answer.replace(question, "").strip()
        answers[question] = cleaned_answer

    return answers




 