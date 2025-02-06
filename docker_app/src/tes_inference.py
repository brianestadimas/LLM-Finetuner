from llms import FinetuneLM
from inference_llms import run_inference_lm
import pandas as pd

# Load sample 101 to 108
train_data = pd.read_csv('train.csv').iloc[99:108]
sample_data = [{'input': row['input'], 'output': row['output']} for _, row in train_data.iterrows()]

for elem in sample_data:
        # Format the data into an array of dictionaries with 'input' and 'output' keys
        user_input = elem["input"] # take input from train _data in i
        
        model_id = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

        # Generate a response
        response = run_inference_lm(user_input=user_input, temperature=1.0, max_tokens=500, model_id=model_id)
        print("INPUT=====================================================================")
        print(user_input)
        print("RESPONSE=====================================================================")
        print(response)
        print("EXPECTED=====================================================================")
        print(elem["output"])
        print("END=====================================================================")