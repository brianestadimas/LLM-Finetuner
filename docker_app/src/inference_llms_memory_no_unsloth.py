from io import BytesIO
import pandas as pd
from unsloth import FastLanguageModel, FastVisionModel
from pdf2image import convert_from_path
from transformers import TextStreamer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file.slides import PptxReader
from llama_index.readers.file.tabular import CSVReader
from llama_index.core.settings import Settings
from llama_index.core.node_parser import TokenTextSplitter
from pathlib import Path
import glob, os, json, re, shutil, time, csv, torch
from src.utils import find_highest_checkpoint, folder_has_files
from PIL import Image
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import base64
from openai import OpenAI

log_file_path = "model_logs.txt"
MODEL = None
TOKENIZER = None
RETRIEVER = None

def initialize_model_no_unsloth(model_id: str, checkpoint_root: str = "./model_cp", separator=" ", chunk_size=4096, chunk_overlap=50, 
                     replace_spaces=False, delete_urls=False, ocr_model='Qwen2.5VL', gpt_api_key=None):
    global MODEL, TOKENIZER, RETRIEVER
    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER, RETRIEVER
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
    recent_history = conversation_history[-8:]
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(recent_history)
    conversation.append({"role": "user", "content": user_input})
    formatted_prompt = TOKENIZER.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt.strip()

def run_inference_lm_memory_no_unsloth(
    model_id,
    user_input,
    conversation_history,
    system_prompt="",
    temperature=0.3,
    max_tokens=1000
):
    """
    1) Let the LLM parse the user_input into a JSON array of chunk(s).
    2) If only 1 chunk, do single-step approach. If multiple, do multi-step.
    """
    model, tokenizer = initialize_model_no_unsloth(model_id)
    FastLanguageModel.for_inference(model)
    
    prompt = format_data_inference(user_input, conversation_history, system_prompt)
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": generated_text})
    
    return generated_text, conversation_history
