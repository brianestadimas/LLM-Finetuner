import os
import torch
from unsloth import FastLanguageModel
from PIL import Image
from unsloth.chat_templates import get_chat_template
from src.utils import find_highest_checkpoint

# Globals for holding the loaded model, tokenizer, and conversation history
MODEL = None
TOKENIZER = None
CONVERSATION_HISTORY = []  # Each entry: {"role": "user"/"assistant", "content": "..."}

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
    except Exception as e:
        print(f"No checkpoint found ({e}), using model_id")
        model_name = model_id

    print(f"Loading model from: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
    )
    
    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER

def update_conversation_history(role: str, content: str):
    """Append a new message to the conversation history."""
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY.append({"role": role, "content": content})

def format_data_inference(tokenizer, conversation_history, model_id: str) -> str:
    """
    Formats the full conversation history as a prompt.
    Uses a chat template if available; otherwise, falls back to a basic format.
    """
    template_name = None
    model_id_lower = model_id.lower()

    if "mistral" in model_id_lower:
        template_name = "mistral"
    elif "llama" in model_id_lower:
        template_name = "llama-3"
    # You can add more conditions (e.g., for deepseek/qwen) as needed.

    if template_name:
        try:
            tokn = get_chat_template(
                tokenizer,
                chat_template=template_name,
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
                map_eos_token=True,
            )
            formatted_text = tokn.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            formatted_text = "\n".join(
                [f"### {msg['role'].capitalize()}:\n{msg['content']}" for msg in conversation_history]
            ) + "\n### Response:\n"
    elif "deepseek" in model_id_lower and "qwen" in model_id_lower:
        formatted_text = "\n".join(
            [f"### {msg['role'].capitalize()}:\n{msg['content']}" for msg in conversation_history]
        ) + "\n### Response:\n"
    else:
        formatted_text = "\n".join(
            [f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" for msg in conversation_history]
        ) + "\n<|im_start|>assistant\n"

    return formatted_text

def run_inference_lm(user_input: str, temperature: float = 1.0, max_tokens: int = 1000,
                     model_id: str = "unsloth/Phi-3.5-mini-instruct") -> str:
    model, tokenizer = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    
    # Add the new user input to the conversation history
    update_conversation_history("user", user_input)
    
    # Format the prompt using the full conversation history
    prompt = format_data_inference(tokenizer, CONVERSATION_HISTORY, model_id)
    
    # Tokenize inputs
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.2,
        use_cache=True 
    )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Optional cleanup for specific models (e.g., Llama)
    if "llama" in model_id.lower():
        unwanted_prefix = "assistant\n\n"
        if generated_text.startswith(unwanted_prefix):
            generated_text = generated_text[len(unwanted_prefix):].lstrip()
    
    # Add the assistant's response to the conversation history
    update_conversation_history("assistant", generated_text)
    
    return generated_text
