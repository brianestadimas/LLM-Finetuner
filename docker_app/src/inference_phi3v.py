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


def initialize_model(checkpoint_root: str = "./model_cp"):
    global MODEL, PROCESSOR

    # If already loaded, just return
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR

    # 1. Base model
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    print("Loading base Phi-3 Vision model...")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,  # Enable 8-bit loading
    #     bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation
    #     bnb_8bit_use_double_quant=True,  # Use double quantization for memory efficiency
    #     device_map="cuda"  # Automatically place on available GPUs
    # )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        # quantization_config=bnb_config,
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

    # 4. Processor
    local_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Processor loaded.")

    # Cache in global variables
    MODEL = lora_model
    PROCESSOR = local_processor

    return MODEL, PROCESSOR


def run_inference(image: Image.Image, user_input: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
    model, processor = initialize_model()

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
