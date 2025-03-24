from unsloth import FastLanguageModel
from transformers import TextStreamer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file.image_caption import ImageCaptionReader
from llama_index.readers.file.image_deplot import ImageTabularChartReader
from llama_index.readers.file.slides import PptxReader
from llama_index.readers.file.tabular import CSVReader
from llama_index.core.settings import Settings
from llama_index.core.node_parser import TokenTextSplitter
from pathlib import Path
import glob, os, json, re
from transformers import TextStreamer
import easyocr
from src.utils import find_highest_checkpoint

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

    print(f"Loading model from: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
    )
    retriever = build_retriever(separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, replace_spaces=replace_spaces, delete_urls=delete_urls)
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

def build_retriever(separator=" ", chunk_size=4096, chunk_overlap=50, replace_spaces=False, delete_urls=False):
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
        docs_url = SimpleWebPageReader().load_data(websites)

    reader = easyocr.Reader(['en'], gpu=True)
    docs_image_caption = []
    for img_path in glob.glob("./src/rags/image_caption/*"):
        result = reader.readtext(img_path)
        recognized_lines = []
        for (bbox, text, confidence) in result:
            recognized_lines.append(text)
        recognized_text = "\n".join(recognized_lines)
        docs_image_caption.append(Document(text=recognized_text, metadata={"source": img_path}))

    image_tabular_reader = ImageTabularChartReader()
    docs_image_table = []
    for chart_file in glob.glob("./src/rags/image_tabular/*"):
        docs_image_table.extend(image_tabular_reader.load_data(chart_file))

    pptx_reader = PptxReader()
    docs_pptx = []
    for pptx_file in glob.glob("./src/rags/pptx/*.pptx"):
        docs_pptx.extend(pptx_reader.load_data(pptx_file))

    csv_reader = CSVReader()
    docs_csv = []
    for csv_file in glob.glob("./src/rags/csv/*.csv"):
        docs_csv.extend(csv_reader.load_data(file=Path(csv_file)))

    # Combine all documents
    docs = (
        docs_local
        + docs_url
        + docs_image_caption
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
