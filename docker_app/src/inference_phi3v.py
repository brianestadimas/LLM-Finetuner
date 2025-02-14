import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from src.utils import find_highest_checkpoint

# Globals for holding the loaded model and processor
MODEL = None
PROCESSOR = None


def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, PROCESSOR

    # If already loaded, just return
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR

    print("Loading base model...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        # quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'
    )
    # 2. Find highest checkpoint
    try:
        adapter_path = find_highest_checkpoint(checkpoint_root)
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except:
        model = base_model
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    MODEL = model
    PROCESSOR = processor
    return MODEL, PROCESSOR


def run_inference_phi3v(image: Image.Image, user_input: str, temperature: float = 0.0, 
                        max_tokens: int = 500, model_id: str = "microsoft/Phi-3-vision-128k-instruct") -> str:

    model, processor = initialize_model(model_id)
    
    # Construct messages for a typical Phi-3 style prompt
    messages = [
        {"role": "user", "content": f"<|image_1|>\n{user_input}"}
    ]
    # Tokenize prompt using the built-in chat template
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Prepare the model inputs
    inputs = processor(
        prompt,
        images=[image],
        return_tensors="pt"
    ).to("cuda")

    # Generation parameters
    generation_args = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": False
    }

    # Generate the output
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # Remove input tokens from the output
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response
