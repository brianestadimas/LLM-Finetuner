from src.utils import find_highest_checkpoint
from src.qwenvl import FinetuneQwenVL
from src.phi3v import FinetunePhi3V
from src.inference_qwenvl import run_inference_qwenvl, run_inference_qwenvl_video
from src.inference_phi3v import run_inference_phi3v
import pandas as pd
from PIL import Image

# Load first 100 sample data from train.csv

sample_data = [{"image": "image.jpg", "input": "What is in the image?", "output": "A bag and only bag"}]

# model_id = "Salesforce/blip2-opt-2.7b"
# model_id = "unsloth/Qwen2-VL-7B-Instruct"
# finetuner = FinetuneQwenVL(data=sample_data, epochs=15, learning_rate=5e-5, model_id=model_id, peft_alpha=16, \
#         peft_r=16, peft_dropout=0.0, gradient_accumulation_steps=8, warmup_ratio=0.1)
# finetuner.run()

## ========================================================================

# del finetuner

# model_id = find_highest_checkpoint("./model_cp")
model_id = "unsloth/Qwen2-VL-7B-Instruct"

# # TEST inference
user_input = "What is the content?"
# image = "image.jpg" # convert to pil
# image_pil = Image.open(image).convert("RGB")
video_path = "Video.mp4"

# Generate a response
response = run_inference_qwenvl_video()
print("INPUT=====================================================================")
print(user_input)
print("RESPONSE=====================================================================")
print(response)