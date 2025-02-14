from src.llms import FinetuneLM
from src.inference_llms import run_inference_lm
import pandas as pd
import re

def inference_llm(user_input, model_id, temperature, max_tokens):
    result = run_inference_lm(user_input, temperature, max_tokens, model_id)
    
    think_match = re.search(r"^(.*?)</think>", result, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        final_result = result[think_match.end():].strip()  # Remove the <think> section
    else:
        think_content = None
        final_result = result.strip()

    response = {"result": final_result}
    if think_content:
        response["think"] = think_content
        
    return response

train_data = pd.read_csv('train.csv').iloc[99:108]
sample_data = [{'input': row['input'], 'output': row['output']} for _, row in train_data.iterrows()]

for elem in sample_data:
    # Format the data into an array of dictionaries with 'input' and 'output' keys
    user_input = elem["input"] # take input from train _data in i
    
    model_id = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

    # Generate a response
    response = inference_llm(user_input=user_input, temperature=1.0, max_tokens=1000, model_id=model_id)
    print("INPUT=====================================================================")
    print(user_input)
    print("RESPONSE=====================================================================")
    print(response)
    print("EXPECTED=====================================================================")
    print(elem["output"])
    print("END=====================================================================")