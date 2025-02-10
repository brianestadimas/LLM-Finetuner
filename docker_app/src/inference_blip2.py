import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Globals for holding the loaded model and processor
MODEL = None
PROCESSOR = None


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
    global MODEL, PROCESSOR

    # If already loaded, just return
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'
    )
    print("Base model loaded.")

    # 2. Find highest checkpoint
    adapter_path = find_highest_checkpoint(checkpoint_root)
    print(f"Highest checkpoint found: {adapter_path}")

    # 3. Load LoRA adapter on top of base model
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    print("LoRA adapter loaded.")

    # 4. Load processor
    local_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Processor loaded.")

    # Cache in global variables
    MODEL = lora_model
    PROCESSOR = local_processor

    return MODEL, PROCESSOR


def run_inference_blip2(
    image: Image.Image, 
    user_input: str, 
    temperature: float = 0.0, 
    max_tokens: int = 500, 
    model_id: str = "Salesforce/blip2-opt-2.7b"
) -> str:
    model, processor = initialize_model(model_id)
    
    # Build a BLIP2-style prompt.
    # You can modify this template as needed.
    prompt = f"Question: {user_input}\nImage: <image>\nAnswer:"
    
    # Prepare the inputs using the processor.
    # For BLIP2, it is common to pass text and images as keyword arguments.
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    # Ensure all tensors are on GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
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
    
    # If the processor produced input_ids, remove them from the output.
    if "input_ids" in inputs:
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    
    # Decode the generated tokens.
    response = processor.tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return response
