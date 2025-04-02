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
import glob, os, json, re
from src.utils import find_highest_checkpoint
from PIL import Image
import torch
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time


log_file_path = "model_logs.txt"
MODEL = None
TOKENIZER = None
RETRIEVER = None

def initialize_model(model_id: str, checkpoint_root: str = "./model_cp", separator=" ", chunk_size=4096, chunk_overlap=50, replace_spaces=False, delete_urls=False):
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

    retriever = build_retriever(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, replace_spaces=replace_spaces, delete_urls=delete_urls)

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
    recent_history = conversation_history[-10:]
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(recent_history)
    conversation.append({"role": "user", "content": user_input})
    formatted_prompt = TOKENIZER.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt.strip()

def load_big_csv_in_chunks(file_path, rows_per_chunk=1000):
    """
    Read the CSV in row batches. For each batch, produce one Document.
    E.g., if you have 200k rows, you'll get ~200 chunks.
    """
    docs = []
    chunk_index = 0
    row_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # If there's a header
        batch = []

        for row in reader:
            batch.append(row)
            row_count += 1

            # If batch hits the limit, log and create a new Document
            if len(batch) >= rows_per_chunk:
                # Log the chunk
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"Processing CSV chunk #{chunk_index} for file {file_path}: "
                        f"rows {row_count - len(batch) + 1} to {row_count}\n"
                    )

                text = convert_rows_to_text(header, batch)
                docs.append(Document(text=text, metadata={"source": file_path}))

                chunk_index += 1
                batch = []

        # leftover rows if batch is not empty
        if batch:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"Processing FINAL CSV chunk #{chunk_index} for file {file_path}: "
                    f"rows {row_count - len(batch) + 1} to {row_count}\n"
                )
            text = convert_rows_to_text(header, batch)
            docs.append(Document(text=text, metadata={"source": file_path}))

    return docs

def convert_rows_to_text(header, rows):
    """
    Turn the chunk of CSV rows into a single string.
    This is your chance to summarize or just join them.
    """
    lines = []
    if header:
        lines.append(",".join(header))
    for r in rows:
        lines.append(",".join(r))
    return "\n".join(lines)

def build_retriever(separator=" ", chunk_size=4096, chunk_overlap=50, replace_spaces=False, delete_urls=False):
    # Load documents from various sources
    try:
        # Replace SimpleDirectoryReader with our Poppler-based loader
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
        chrome_options.add_argument("--headless")  # Optional: remove for visible browser
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        for url in websites:
            # Remove any trailing slash
            url = url.rstrip('/')
            success = False
            attempts = 0
            max_attempts = 3

            while not success and attempts < max_attempts:
                attempts += 1
                try:
                    driver.get(url)
                    # Wait until the body element is present (up to 10 seconds)
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    body = driver.find_element(By.TAG_NAME, "body")
                    text = body.text.strip()
                    if text:
                        docs_url.append(Document(text=text, metadata={"source": url}))
                    success = True
                except Exception as e:
                    print(f"Attempt {attempts} failed to load {url}: {e}")
                    if attempts < max_attempts:
                        time.sleep(2)  # Wait 2 seconds before retrying

            if not success:
                print(f"Failed to load {url} after {max_attempts} attempts.")

        driver.quit()

    model_id = "unsloth/Qwen2.5-VL-7B-Instruct"
    model, tokenizer = FastVisionModel.from_pretrained(model_id, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    
    def extract_image_docs(folder, prompt_text):
        docs = []
        for path in glob.glob(f"{folder}/*"):
            image = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(image, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=2048, temperature=0.0,
                    do_sample=False, min_p=0.1, use_cache=True
                )
            result = tokenizer.decode(
                output_ids[:, inputs["input_ids"].shape[1]:][0],
                skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            docs.append(Document(text=result, metadata={"source": path}))
        return docs
    
    def extract_pdf_ocr_docs(folder, prompt_text):
        all_docs = []
        for pdf_path in glob.glob(f"{folder}/*.pdf"):
            try:
                pages = convert_from_path(pdf_path)  # Convert PDF to list of PIL pages
            except Exception as e:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Failed to convert {pdf_path}: {e}")
                continue

            for i, page_img in enumerate(pages):
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Processing OCR on page {i}, {pdf_path}\n")
                page_img = page_img.convert("RGB")
                page_img = page_img.resize(
                    (page_img.width // 2, page_img.height // 2),
                    resample=Image.LANCZOS
                )
                
                # Build the prompt
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
                prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True)
                inputs = tokenizer(page_img, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False, min_p=0.1)
                text = tokenizer.decode(
                    out_ids[:, inputs["input_ids"].shape[1]:][0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()
                all_docs.append(Document(text=text, metadata={"source": pdf_path, "page": i}))
        return all_docs

    docs_image_caption = extract_image_docs(
        "./src/rags/image_caption",
        "Please extract all the text from this image, as detail as possible."
    )
    
    docs_image_description = extract_image_docs(
        "./src/rags/image_desc",
        "Describe the image in detail."
    )

    docs_image_table = extract_image_docs(
        "./src/rags/image_tabular",
        "Please extract all tabular or chart information from this image (also small description what is chart/tabular about), as detailed and structured as possible including numbers if available."
    )
    
    docs_pdf_ocr = extract_pdf_ocr_docs(
        "./src/rags/pdf_ocr",
        "Please extract all text and tabular/chart from this screenshot page, as detailed and structured as possible including numbers if available. Pay close attention to the layout and positioning: preserve the top-to-bottom and left-to-right reading order as it appears visually. For tables, clearly align each value with the correct row and column headers. Include numerical data accurately and structure the output in a readable format such as markdown tables or clearly separated sections."
    )

    model = model.cpu()
    model = None
    del model
    del tokenizer
    with torch.no_grad():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    pptx_reader = PptxReader()
    docs_pptx = []
    for pptx_file in glob.glob("./src/rags/pptx/*.pptx"):
        docs_pptx.extend(pptx_reader.load_data(pptx_file))

    docs_csv = []
    for csv_file in glob.glob("./src/rags/csv/*.csv"):
        docs_csv.extend(load_big_csv_in_chunks(csv_file, rows_per_chunk=2000))

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
    # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device="cuda")
    # embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    # embed_model = HuggingFaceEmbedding(
    #     model_name="Linq-AI-Research/Linq-Embed-Mistral",
    #     device="cuda",
    #     trust_remote_code=True,
    #     model_kwargs={"load_in_4bit": True}
    # )

    
    # Service context with text splitter
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter

    # Build index with chunked documents
    index = VectorStoreIndex.from_documents(
        chunked_docs,
        storage_context=sc,
    )
    return index.as_retriever()

def retrieve_context(user_input, retriever, top_k=3):
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
        f"Please use this context to respond accurately.\n"
    )
    
    # Format prompt with conversation history
    prompt = format_data_inference(user_input, conversation_history, rag_prompt)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=4096
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.1,
        use_cache=True
    )
    
    # Decode and format response
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
        repetition_penalty=1.1,
        use_cache=True
    )

    raw_parse = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    try:
        parsed = json.loads(raw_parse)
        # Must ensure it's a list of strings
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
        else:
            return [user_input]
    except:
        return [user_input]
    
def run_inference_lm_memory(
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
    model, tokenizer, retriever = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    
    # Parse user input with the LLM
    chunks = parse_user_input_with_llm(user_input, model_id)
    
    # If only 1 chunk => single-step
    if len(chunks) == 1:
        return run_inference_lm_memory_with_rag_single(
            model_id,
            user_input,
            conversation_history,
            system_prompt,
            model,
            tokenizer,
            retriever,
            temperature,
            max_tokens
        )
    all_responses = []
    for chunk in chunks:
        # -- same sanitization to avoid quotes in LanceDB query
        sanitized_chunk = chunk.replace('"', '').replace(',', ' ')
        retrieved = retrieve_context(sanitized_chunk, retriever, top_k=3)
        
        rag_prompt = (
            f"{system_prompt}\n"
            f"User request:\n{chunk}\n"
            f"Here is some relevant retrieved context:\n{retrieved}\n\n"
            f"Please use this context to respond accurately.\n"
        )
        
        prompt = format_data_inference(chunk, conversation_history, rag_prompt)
        
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
            repetition_penalty=1.1,
            use_cache=True
        )
        
        gen_sub = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        conversation_history.append({"role": "user", "content": chunk})
        conversation_history.append({"role": "assistant", "content": gen_sub})
        
        all_responses.append(f"{chunk}\n{gen_sub}")

    # Join partial answers with a blank line
    final_answer = "\n\n".join(all_responses)
    return final_answer, conversation_history


def apply_text_preprocessing(text: str, replace_spaces: bool, delete_urls: bool) -> str:
    """Apply text cleanup rules if requested."""
    if replace_spaces:
        # Replace consecutive whitespace (spaces, newlines, tabs) with a single space
        text = re.sub(r"\s+", " ", text)

    if delete_urls:
        # Remove URLs, emails, etc. 
        # For URLs, something like:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # For emails, if you want that too:
        text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Optionally trim leading/trailing spaces
    text = text.strip()
    return text