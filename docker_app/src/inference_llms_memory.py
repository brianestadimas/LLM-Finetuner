from unsloth import FastLanguageModel
from transformers import TextStreamer
from utils import find_highest_checkpoint

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
        load_in_4bit=False,
    )
    
    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER

def format_data_inference(user_input, conversation_history, system_prompt):
    recent_history = conversation_history[-10:]
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(recent_history)
    conversation.append({"role": "user", "content": user_input})
    formatted_prompt = TOKENIZER.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt.strip()

def run_inference_lm_memory(model_id, user_input, conversation_history, system_prompt, temperature=0.7, max_tokens=500):
    model, tokenizer = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    
    prompt = format_data_inference(user_input, conversation_history, system_prompt)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        use_cache=True,
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    ).strip()

    updated_conversation_history = conversation_history
    updated_conversation_history.append({"role": "user", "content": user_input})
    updated_conversation_history.append({"role": "assistant", "content": generated_text})

    return generated_text, updated_conversation_history