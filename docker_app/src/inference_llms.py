import os
import torch
from unsloth import FastLanguageModel
from PIL import Image
from unsloth.chat_templates import get_chat_template

# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None


def find_highest_checkpoint(checkpoint_dir: str) -> str:
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by the numeric portion after "checkpoint-"
    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    highest_checkpoint = checkpoints_sorted[-1]
    return os.path.join(checkpoint_dir, highest_checkpoint)


def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, TOKENIZER

    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    adapter_path = find_highest_checkpoint(checkpoint_root)
    print(f"Highest checkpoint found: {adapter_path}")
    
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name =  adapter_path,  # Trained model either locally or from huggingface
        load_in_4bit = False,
    )
    print("Base model loaded.")
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.special_tokens_map["eos_token"] = "<|im_end|>"

    # 2. Find highest checkpoint

    MODEL = model
    TOKENIZER = tokenizer

    return MODEL, TOKENIZER

# def format_data(tokenizer, user_input):
#     messages = [
#         {"role": "user", "content": user_input},
#     ]
#     try:
#         formatted_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False,
#         )
#     except Exception as e:
#         formatted_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>"
    
#     return formatted_text

def format_data(tokenizer, user_input):
    """
    Formats the user input into the Qwen chat template format.
    """
    formatted_text = (
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return formatted_text


def run_inference_lm(user_input: str, temperature: float = 1.0, max_tokens: int = 1000, model_id: str = "unsloth/Phi-3.5-mini-instruct") -> str:

    model, tokenizer = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    prompt = format_data(tokenizer, user_input) 

    # 4. Tokenize inputs
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")
    

    # 5. Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        # pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2,
        use_cache=True 
    )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    return generated_text