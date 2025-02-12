from qwenvl import FinetuneQwenVL
from phi3v import FinetunePhi3V
from inference_qwenvl import run_inference_qwenvl
from inference_phi3v import run_inference_phi3v
import pandas as pd
from PIL import Image

# Load first 100 sample data from train.csv
train_data = pd.read_csv('train.csv').head(150)

sample_data = []

model_id = "microsoft/Phi-3-vision-128k-instruct"
finetuner = FinetunePhi3V(data=sample_data, epochs=1, learning_rate=5e-6, model_id=model_id, peft_alpha=16, \
        peft_r=16, peft_dropout=0.0, gradient_accumulation_steps=8, warmup_ratio=0.1)
finetuner.run()

# TEST inference
user_input = "What is in the image?"
image = "image.jpg" # convert to pil
image_pil = Image.open(image).convert("RGB")

# Generate a response
response = run_inference_phi3v(image=image_pil, user_input=user_input, temperature=1.0, max_tokens=500, model_id=model_id)
print("INPUT=====================================================================")
print(user_input)
print("RESPONSE=====================================================================")
print(response)