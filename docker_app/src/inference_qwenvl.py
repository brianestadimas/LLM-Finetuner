import os
import torch
from unsloth import FastVisionModel
from PIL import Image
from src.utils import find_highest_checkpoint
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None

def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, TOKENIZER
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    try:
        adapter_path = find_highest_checkpoint(checkpoint_root)
        print(f"Highest checkpoint found: {adapter_path}")
        model_name = adapter_path
    except:
        model_name = model_id
    
    print("Loading base model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name =  model_name,  # Trained model either locally or from huggingface
        load_in_4bit = True,
    )
    print("Base model loaded.")
    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER

def run_inference_qwenvl(image: Image.Image, user_input: str, temperature: float = 0.0, 
                        max_tokens: int = 500, model_id: str = "unsloth/Qwen2-VL-7B-Instruct") -> str:

    model, tokenizer = initialize_model(model_id)
    FastVisionModel.for_inference(model) 
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": user_input
                },
            ]
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature,
        do_sample=False,
        min_p=0.1
    )
    generate_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return generated_text


def run_inference_qwenvl(image, user_input, temperature=0.0, max_tokens=500, model_id="unsloth/Qwen2-VL-7B-Instruct"):
    model, tokenizer = initialize_model(model_id)
    FastVisionModel.for_inference(model)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_input}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, temperature=temperature, do_sample=False, min_p=0.1)
    
    return tokenizer.decode(output_ids[:, inputs['input_ids'].shape[1]:][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run_inference_qwenvl_video(video_path="./Video.mp4", user_input="Describe video", temperature=0.0, 
                               max_tokens=500, fps=1, model_id="unsloth/Qwen2-VL-7B-Instruct") -> str:
    model, tokenizer = initialize_model(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    FastVisionModel.for_inference(model)

    messages = [{"role": "user", "content": [{"type": "video", "video": f"{video_path}", "max_pixels": 360 * 420, "fps": fps}, 
                                             {"type": "text", "text": user_input}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, temperature=temperature, do_sample=False, min_p=0.1, repetition_penalty=1.1,)

    return processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]