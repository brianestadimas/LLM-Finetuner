from unsloth import FastLanguageModel
from transformers import TextStreamer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file.image_caption import ImageCaptionReader
from llama_index.readers.file.image_deplot import ImageTabularChartReader
from llama_index.readers.file.slides import PptxReader
from llama_index.readers.file.tabular import CSVReader
from pathlib import Path
import glob, os
from transformers import TextStreamer
from src.utils import find_highest_checkpoint

MODEL = None
TOKENIZER = None
RETRIEVER = None

def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
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
    retriever = build_retriever()
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

def build_retriever():
    docs_local = SimpleDirectoryReader("./src/rags/pdf").load_data()

    websites = []
    website_txt = "./src/rags/website.txt"
    if os.path.exists(website_txt):
        with open(website_txt, "r", encoding="utf-8") as f:
            websites = [line.strip() for line in f if line.strip()]

    docs_url = []
    if websites:
        docs_url = SimpleWebPageReader().load_data(websites)

    image_caption_reader = ImageCaptionReader()
    docs_image_caption = []
    for img_file in glob.glob("./src/rags/image_caption/*"):
        docs_image_caption.extend(image_caption_reader.load_data(img_file))

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

    docs = (
        docs_local
        + docs_url
        + docs_image_caption
        + docs_image_table
        + docs_pptx
        + docs_csv
    )

    vs = LanceDBVectorStore(uri="./lancedb", mode="overwrite", query_type="vector")
    sc = StorageContext.from_defaults(vector_store=vs)
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large", device="cuda")

    index = VectorStoreIndex.from_documents(docs, storage_context=sc, embed_model=embed_model)
    return index.as_retriever()

def retrieve_context(user_input, retriever, top_k=5):
    docs = retriever.retrieve(user_input)
    return "\n\n".join(doc.text for doc in docs[:top_k])

def run_inference_lm_memory(model_id, user_input, conversation_history, system_prompt, temperature=0.5, max_tokens=500):
    model, tokenizer, retriever = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    retrieved = retrieve_context(user_input, retriever).strip()
    if retrieved:
        rag_prompt = (
            f"{system_prompt}\n\nHere is some relevant retrieved context:\n{retrieved}\n\n"
            f"Please use this context to answer accurately.\n"
        )
    else:
        rag_prompt = system_prompt

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