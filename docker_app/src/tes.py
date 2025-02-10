from llms import FinetuneLM
from inference_llms import run_inference_lm
import pandas as pd

# Load first 100 sample data from train.csv
train_data = pd.read_csv('train.csv').head(10)

# Format the data into an array of dictionaries with 'input' and 'output' keys
sample_data = [{'input': row['input'], 'output': row['output']} for index, row in train_data.iterrows()]

model_id = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
finetuner = FinetuneLM(data=sample_data, epochs=2, learning_rate=5e-6, model_id=model_id, peft_alpha=16, \
        peft_r=15, peft_dropout=0.0, gradient_accumulation_steps=8, warmup_ratio=0.1)
finetuner.run()