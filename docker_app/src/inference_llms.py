import os
import threading
import torch
from unsloth import FastLanguageModel
from PIL import Image
from unsloth.chat_templates import get_chat_template
from src.utils import find_highest_checkpoint
from transformers import TextIteratorStreamer

# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None


def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, TOKENIZER

    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    # Check if local fine-tuned model is present and non-empty
    try:
        adapter_path = find_highest_checkpoint(checkpoint_root)
        print(f"Highest checkpoint found: {adapter_path}")
        model_name = adapter_path
    except:
        model_name = model_id

    print(f"Loading model from: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
    )
    
    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER


def format_data_inference(tokenizer, user_input, model_id: str) -> str:
    template_name = None
    model_id_lower = model_id.lower()

    if "mistral" in model_id_lower:
        template_name = "mistral"
    elif "llama" in model_id_lower:
        template_name = "llama-3"
    # elif "deepseek" in model_id_lower and "qwen" in model_id_lower:
    #     template_name = None

    if template_name:
        row_json = [{"role": "user", "content": user_input}]
        tokn = get_chat_template(
            tokenizer,
            chat_template=template_name,
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            map_eos_token=True,
        )
        try:
            formatted_text = tokn.apply_chat_template(
                row_json,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            formatted_text = f"### Instruction:\n{user_input}\n### Response:\n"
    elif "deepseek" in model_id_lower and "qwen" in model_id_lower:
        formatted_text = f"### Instruction:\n{user_input}\n### Response:\n"
    else:
        formatted_text = (
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return formatted_text


def run_inference_lm(user_input: str, temperature: float = 1.0, max_tokens: int = 1000, model_id: str = "unsloth/Phi-3.5-mini-instruct") -> str:
    model, tokenizer = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    prompt = format_data_inference(tokenizer, user_input, model_id) 

    # 4. Tokenize inputs
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

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
    if "llama" in model_id.lower():
        unwanted_prefix = "assistant\n\n"
        if generated_text.startswith(unwanted_prefix):
            generated_text = generated_text[len(unwanted_prefix):].lstrip()
    
    return generated_text


def run_inference_lm_streaming(
    user_input: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    model_id: str = "unsloth/Phi-3.5-mini-instruct"
):
    model, tokenizer = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    prompt = format_data_inference(tokenizer, user_input, model_id)

    # 1. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 2. Create a TextIteratorStreamer to capture generated tokens
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,           # so we only stream new tokens
        skip_special_tokens=True    # omit special tokens
    )

    # 3. Launch generation in a background thread so we can yield tokens
    def generate_in_background():
        model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False,
            repetition_penalty=1.2,
            use_cache=True,
            streamer=streamer  # <-- The magic: streamer receives tokens as they arrive
        )

    thread = threading.Thread(target=generate_in_background)
    thread.start()

    # 4. Yield tokens from the streamer iterator, one by one
    for new_text in streamer:
        # This is just raw text. You could chunk or accumulate if you prefer.
        yield new_text

    thread.join()