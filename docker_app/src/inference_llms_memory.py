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
import threading
from transformers import TextIteratorStreamer


log_file_path = "model_logs.txt"
MODEL = None
TOKENIZER = None
RETRIEVER = None

PDF_OCR_PROMPT_TEXT = (
"""You are an OCR tool. Your task is to extract and structure the content of this page into two separate sections:
1. Original Context:
Extract all visible text, chart, and numbers exactly as they appear on the page. Do not summarize or interpret. Preserve the original reading order and formatting.
2. Layout Description:
Describe the layout, chart positioning with it numbers, and structure of the page. Include information on:
- Be extra careful with statistics numbers, dont mix up
- Table positions, rows, and columns (be careful with numbers)
- Chart types, legends, axes, and color usage
- The relative placement of key sections (top-left, center, footer, etc.)
- Any coloring used to differentiate groups or emphasize data

Important: Do NOT add any analysis, opinion, or summary. Just extract exactly what is shown on the page and describe the layout objectively."""
)

"Describe the image in detail."

IMAGE_OCR_PROMPT_TEXT = (
"""You are an OCR tool for image/form. Your task is to extract and structure the content of the image, extract all text, including handwritten text, filled in text, numbers, and text inside boxes.
Important: Do NOT add any analysis, opinion, or summary. Just extract exactly what is shown on the page in RAW. Be extra careful with text inside boxes, especially if color is gray"""
)

CHART_OCR_PROMPT_TEXT = (
"""You are an OCR tool. Your task is to extract and structure the content of this image contain table/chart/graph.
(IF chart/graph) put into two separate sections:
1. Original Context:
Extract all visible text, chart, and numbers exactly as they appear on the page. Do not summarize or interpret. Preserve the original reading order and formatting.
2. Layout Description:
Describe the layout, chart positioning with it numbers, and structure of the page. Include information on:
- Be extra careful with statistics numbers, dont mix up
- Table positions, rows, and columns (be careful with numbers)
- Chart types, legends, axes, and color usage
- The relative placement of key sections (top-left, center, footer, etc.)
- Any coloring used to differentiate groups or emphasize data
- (If radial chart) Extract numbers of chart in extra careful, small fonts

(IF table) extract the table originally in table format, be extra careful with rows and columns, and value inside cells.

Important: Do NOT add any analysis, opinion, or summary. Just extract exactly what is shown on the page and describe the layout objectively."""
)

TABLE_OCR_PROMPT_TEXT = (
"""You are an OCR tool. The image provided is a table. Extract the table in **original table format**, preserving the structure.
Be extra careful with:
- Each column's alignment
- Row data and labels
- Values inside each cell, especially `/` vs empty spaces
- Do NOT summarize or interpret. Just extract what's shown.
Output must strictly be in table format (Markdown or CSV), with clear column and row alignment."""
)

IMAGE_DESC_PROMPT_TEXT = (
"Describe the image in detail."
)

def initialize_model(model_id: str, checkpoint_root: str = "./model_cp", separator=" ", chunk_size=4096, chunk_overlap=50, 
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

    retriever = build_retriever(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, replace_spaces=replace_spaces, 
                                delete_urls=delete_urls, ocr_model=ocr_model, gpt_api_key=gpt_api_key)

    print(f"Loading model from: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
    )
    RETRIEVER = retriever
    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER, retriever

def format_data_inference(user_input, conversation_history, system_prompt):
    recent_history = conversation_history[-8:]
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(recent_history)
    conversation.append({"role": "user", "content": user_input})
    formatted_prompt = TOKENIZER.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt.strip()

def build_retriever(separator=" ", chunk_size=4096, chunk_overlap=50, replace_spaces=False, delete_urls=False, ocr_model='Qwen2.5VL', gpt_api_key=None):
    # Load documents from various sources
    try:
        docs_local = SimpleDirectoryReader("./src/rags/pdf").load_data()
    except:
        docs_local = []

    websites = []
    website_txt = "./src/rags/website.txt"
    if os.path.exists(website_txt):
        with open(website_txt, "r", encoding="utf-8") as f:
            websites = [line.strip() for line in f if line.strip()]

    docs_url = []
    if websites:
        chrome_options = Options()
        chrome_options.binary_location = "/usr/bin/google-chrome"
        chrome_options.add_argument("--headless=new")  # Modern headless Chrome
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--ignore-certificate-errors")   # <--- add
        chrome_options.add_argument("--ignore-ssl-errors")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                                    " AppleWebKit/537.36 (KHTML, like Gecko)"
                                    " Chrome/122.0.0.0 Safari/537.36")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        for url in websites:
            url = url.rstrip('/')
            success = False
            attempts = 0
            max_attempts = 2

            while not success and attempts < max_attempts:
                attempts += 1
                try:
                    driver.get(url)

                    # Wait for the document to fully load
                    WebDriverWait(driver, 10).until(
                        lambda d: d.execute_script("return document.readyState") == "complete"
                    )
                    WebDriverWait(driver, 10).until(
                        lambda d: len(d.find_element(By.TAG_NAME, "body").text.strip()) > 100
                    )
                    body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
                    docs_url.append(Document(text=body_text, metadata={"source": url}))
                    success = True

                except Exception as e:
                    print(f"[Attempt {attempts}/{max_attempts}] Error loading {url}: {e}")
                    time.sleep(3)  # Increase sleep to give the site time to settle before retrying

            if not success:
                print(f"Failed to capture text from {url} with Selenium; trying other methods...")

                import requests
                from bs4 import BeautifulSoup

                try:
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/122.0.0.0 Safari/537.36"
                        )
                    }
                    r = requests.get(url, headers=headers, timeout=15)
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, "html.parser")
                        fallback_text = soup.get_text("\n", strip=True)
                        docs_url.append(Document(text=fallback_text, metadata={"source": url}))
                        print(f"Fallback success: captured text from {url}")
                    else:
                        print(f"Fallback request returned code {r.status_code} for {url}")
                except Exception as e2:
                    print(f"Requests fallback also failed for {url}: {e2}")

        driver.quit()
    
    def extract_docs(folder, prompt_text, doc_type="image", ocr_model='Qwen2.5VL'):
        docs = []
        if ocr_model.lower() == "gpt-4o-mini":
            client = OpenAI(api_key=gpt_api_key)
            if doc_type == "image":
                for path in glob.glob(f"{folder}/*"):
                    with open(path, "rb") as f:
                        base64_img = base64.b64encode(f.read()).decode("utf-8")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt_text},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{base64_img}"},
                            ],
                        }
                    ]
                    response = client.responses.create(model="gpt-4o-mini", input=messages)
                    docs.append(Document(text=response.output_text, metadata={"source": path}))
            elif doc_type == "pdf":
                for pdf_path in glob.glob(f"{folder}/*.pdf"):
                    try:
                        pages = convert_from_path(pdf_path)
                    except:
                        continue
                    for i, page_img in enumerate(pages):
                        page_img = page_img.convert("RGB").resize(
                            (page_img.width // 2, page_img.height // 2), Image.LANCZOS
                        )
                        buf = BytesIO()
                        page_img.save(buf, format="PNG")
                        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt_text},
                                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_img}"},
                                ],
                            }
                        ]
                        response = client.responses.create(model="gpt-4o-mini", input=messages)
                        docs.append(Document(text=response.output_text, metadata={"source": pdf_path, "page": i}))
                        
        else:
            if doc_type == "image":
                for path in glob.glob(f"{folder}/*"):
                    image = Image.open(path).convert("RGB")
                    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
                    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = tokenizer(image, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            temperature=0.0,
                            do_sample=False,
                            min_p=0.1,
                            use_cache=True
                        )
                    result = tokenizer.decode(
                        output_ids[:, inputs["input_ids"].shape[1]:][0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    ).strip()
                    docs.append(Document(text=result, metadata={"source": path}))
            elif doc_type == "pdf":
                for pdf_path in glob.glob(f"{folder}/*.pdf"):
                    try:
                        pages = convert_from_path(pdf_path)
                    except Exception as e:
                        with open(log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"Failed to convert {pdf_path}: {e}")
                        continue
                    for i, page_img in enumerate(pages):
                        with open(log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"Processing OCR on page {i}, {pdf_path}\n")
                        page_img = page_img.convert("RGB")
                        page_img = page_img.resize((page_img.width // 2, page_img.height // 2), resample=Image.LANCZOS)
                        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
                        prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True)
                        inputs = tokenizer(page_img, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                        with torch.no_grad():
                            out_ids = model.generate(
                                **inputs,
                                max_new_tokens=2048,
                                temperature=0.0,
                                do_sample=False,
                                min_p=0.1
                            )
                        text = tokenizer.decode(
                            out_ids[:, inputs["input_ids"].shape[1]:][0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        ).strip()
                        docs.append(Document(text=text, metadata={"source": pdf_path, "page": i}))
        return docs

    folder_caption = "./src/rags/image_caption"
    folder_desc = "./src/rags/image_desc"
    folder_tabular = "./src/rags/image_tabular"
    folder_pdf_ocr = "./src/rags/pdf_ocr"
    
    if (
        folder_has_files(folder_caption)
        or folder_has_files(folder_desc)
        or folder_has_files(folder_tabular)
        or folder_has_files(folder_pdf_ocr)
    ):
        # model_id="unsloth/granite-vision-3.2-2b-unsloth-bnb-4bit"
        if ocr_model == "Qwen2.5VL":
            model_id = "unsloth/Qwen2.5-VL-7B-Instruct"
            model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=True)
            FastVisionModel.for_inference(model)
        else:
            model, tokenizer = None, None

        docs_image_caption = extract_docs(
            folder_caption,
            IMAGE_OCR_PROMPT_TEXT,
            "image",
            ocr_model
        )
    
        docs_image_description = extract_docs(
            folder_desc,
            IMAGE_DESC_PROMPT_TEXT,
            "image",
            ocr_model
        )

        docs_image_table = extract_docs(
            folder_tabular,
            CHART_OCR_PROMPT_TEXT,
            "image",
            ocr_model
        )

        docs_pdf_ocr = extract_docs(
            folder_pdf_ocr,
            PDF_OCR_PROMPT_TEXT,
            "pdf",
            ocr_model
        )
        
        if model is not None:
            model = model.cpu()
            model = None
            del model
            del tokenizer
        with torch.no_grad():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # hf_cache_path = os.path.expanduser("~/.cache/huggingface")
        # if os.path.exists(hf_cache_path):
        #     shutil.rmtree(hf_cache_path)

    else:
        docs_image_caption = []
        docs_image_description = []
        docs_image_table = []
        docs_pdf_ocr = []

    pptx_reader = PptxReader()
    docs_pptx = []
    for pptx_file in glob.glob("./src/rags/pptx/*.pptx"):
        docs_pptx.extend(pptx_reader.load_data(pptx_file))

    docs_csv = []
    try:
        docs_csv = SimpleDirectoryReader("./src/rags/csv").load_data()
    except:
        docs_csv = []

    # Combine all documents
    docs = (
        docs_local
        + docs_pdf_ocr
        + docs_url
        + docs_image_caption
        + docs_image_description
        + docs_image_table
        + docs_pptx
        + docs_csv
    )

    # Text splitting configuration
    text_splitter = TokenTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        backup_separators=["\n", "."]
    )

    # Split documents into chunks
    chunked_docs = []
    for doc in docs:
        cleaned_text = apply_text_preprocessing(
            doc.text,
            replace_spaces=replace_spaces,
            delete_urls=delete_urls
        )
        
        chunks = text_splitter.split_text(cleaned_text)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                text=chunk,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": i,
                    "original_length": len(doc.text.split())
                }
            ))

    # Vector store configuration
    vs = LanceDBVectorStore(uri="./lancedb", mode="overwrite", query_type="hybrid")
    sc = StorageContext.from_defaults(vector_store=vs)
    
    # Embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device="cuda")
    # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    # embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large", device="cuda")
    
    # Service context with text splitter
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter

    # Build index with chunked documents
    index = VectorStoreIndex.from_documents(
        chunked_docs,
        storage_context=sc,
    )
    return index.as_retriever()

def retrieve_context(user_input, retriever, top_k=2):
    # ---- Simple sanitization to avoid LanceDB FTS syntax errors ----
    sanitized_input = user_input.replace('"', '').replace(',', ' ')
    
    docs = retriever.retrieve(sanitized_input)
    # Filter and sort documents based on relevance
    filtered_docs = sorted(docs[:top_k], key=lambda x: x.score if x.score else 0, reverse=True)
    return "\n\n".join(doc.text for doc in filtered_docs)

def run_inference_lm_memory_with_rag_single(
    model_id,
    user_input,
    conversation_history,
    system_prompt,
    model,
    tokenizer,
    retriever,
    temperature=0.3,
    max_tokens=1000
):
    retrieved = retrieve_context(user_input, retriever)
    
    # Construct RAG prompt
    rag_prompt = (
        f"{system_prompt}\n"
        f"User request:\n{user_input}\n"
        f"Here is some relevant retrieved context:\n{retrieved}\n\n"
        f"Please use this context to respond accurately and try to be strict/objective.\n"
    )
    
    prompt = format_data_inference(user_input, conversation_history, rag_prompt)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.2,
        use_cache=True
    )
    
    gen = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": gen})
    
    return gen, conversation_history

def parse_user_input_with_llm(user_input: str, model_id: str):
    model, tokenizer, retriever = initialize_model(model_id)
    FastLanguageModel.for_inference(model)

    system_prompt = (
        "You are a text parser. You must read the entire user input below and decide:\n"
        " - If it is multiple separate queries or instructions, split them into multiple elements.\n"
        " - If it is a single question or statement (including possible line breaks), keep it as one.\n"
        "\n"
        "Return your answer *strictly* as valid JSON array of strings, e.g.:\n"
        " [\"(first chunk)\", \"(second chunk)\"]\n"
        "\n"
        "No additional commentary or keys. *Only* output the JSON array.\n"
        "Make sure to properly escape any quotes within the array."
    )
    user_prompt = f"USER INPUT:\n{user_input}\n\nProduce the JSON array now."

    parse_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        parse_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=False,
        repetition_penalty=1.2,
        use_cache=True
    )

    raw_parse = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    if not raw_parse:
        return [user_input]
    
    try:
        parsed = json.loads(raw_parse)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
        else:
            return [user_input]
    except:
        return [user_input]
    

def apply_text_preprocessing(text: str, replace_spaces: bool, delete_urls: bool) -> str:
    """Apply text cleanup rules if requested."""
    if replace_spaces:
        text = re.sub(r"\s+", " ", text)

    if delete_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)

    text = text.strip()
    return text
    
def build_rag_prompt(system_prompt, chunk, retrieved):
    return (
        f"{system_prompt}\n"
        f"User request:\n{chunk}\n"
        f"Here is some relevant retrieved context:\n{retrieved}\n\n"
        f"Please use this context to respond accurately and try to be strict/objective.\n"
    )

def generate_response_text(model, tokenizer, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.2,
        use_cache=True
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def run_inference_lm_memory(model_id, user_input, conversation_history, system_prompt="", temperature=0.3, max_tokens=1000):
    model, tokenizer, retriever = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    chunks = parse_user_input_with_llm(user_input, model_id)
    if len(chunks) == 1:
        return run_inference_lm_memory_with_rag_single(
            model_id, user_input, conversation_history, system_prompt, model, tokenizer, retriever, temperature, max_tokens
        )
    all_responses = []
    for chunk in chunks:
        retrieved = retrieve_context(chunk.replace('"', '').replace(',', ' '), retriever)
        prompt = format_data_inference(chunk, conversation_history, build_rag_prompt(system_prompt, chunk, retrieved))
        response = generate_response_text(model, tokenizer, prompt, max_tokens, temperature)
        conversation_history.append({"role": "user", "content": chunk})
        conversation_history.append({"role": "assistant", "content": response})
        all_responses.append(f"{chunk}\n{response}")
    return "\n\n".join(all_responses), conversation_history


def run_inference_lm_streaming_memory(model_id, user_input, conversation_history, system_prompt="", temperature=0.3, max_tokens=1000):
    model, tokenizer, retriever = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    chunks = parse_user_input_with_llm(user_input, model_id)
    if len(chunks) == 1:
        response, history = run_inference_lm_memory_with_rag_single(
            model_id, user_input, conversation_history, system_prompt, model, tokenizer, retriever, temperature, max_tokens
        )
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        yield response
        return
    for chunk in chunks:
        retrieved = retrieve_context(chunk.replace('"', '').replace(',', ' '), retriever)
        prompt = format_data_inference(chunk, conversation_history, build_rag_prompt(system_prompt, chunk, retrieved))
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_special_tokens=True, skip_prompt=True)
        def generate_in_background():
            model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=False,
                repetition_penalty=1.2,
                use_cache=True,
                streamer=streamer
            )
        thread = threading.Thread(target=generate_in_background)
        thread.start()
        generated_text = ""
        for token in streamer:
            generated_text += token
            yield token
        thread.join()
        conversation_history.append({"role": "user", "content": chunk})
        conversation_history.append({"role": "assistant", "content": generated_text})
        
