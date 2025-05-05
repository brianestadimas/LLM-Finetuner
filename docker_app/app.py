import uuid
from flask import Flask, Response, jsonify, request, send_file
import requests
import torch
from src.phi3v import FinetunePhi3V
from src.blip2 import FinetuneBLIP2
from src.qwenvl import FinetuneQwenVL
from src.llms import FinetuneLM, olive_opt
from src.llms_multiturn import FinetuneLMAgent
from flask_cors import CORS
import os, json, sys, logging, re, base64, time, threading, io
from typing import List
from PIL import Image
from src.inference_phi3v import run_inference_phi3v
from src.inference_qwenvl import run_inference_qwenvl, run_inference_qwenvl_video
from src.inference_llms import run_inference_lm, run_inference_lm_streaming
from src.inference_llms_memory import run_inference_lm_memory, initialize_model, run_inference_lm_streaming_memory
from src.inference_llms_memory_no_unsloth import run_inference_lm_memory_no_unsloth, initialize_model_no_unsloth, run_inference_lm_streaming_memory_no_unsloth
from werkzeug.serving import WSGIRequestHandler
from werkzeug.utils import secure_filename
from src.db import init_db, SessionLocal, Conversation, ensure_default_session, DEFAULT_SESSION_ID
from src.utils import save_history, load_history

WSGIRequestHandler.protocol_version = "HTTP/1.1"

app = Flask(__name__)
init_db()
ensure_default_session()
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables to manage the finetuning process
is_running = False
finetune_thread = None

log_file_path = "model_logs.txt"

class ExcludeAPILoggingFilter(logging.Filter):
    def filter(self, record):
        # Define patterns to exclude
        exclude_patterns = [
            r"GET /current_logs HTTP/1.1",
            r"POST /run_model HTTP/1.1",
            r"GET /logs HTTP/1.1",
        ]
        log_message = record.getMessage()
        # Return False if the log message matches any exclude pattern
        return not any(re.search(pattern, log_message) for pattern in exclude_patterns)

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
logger = logging.getLogger()
logger.addFilter(ExcludeAPILoggingFilter())

log_file = open(log_file_path, "a", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

flask_logger = logging.getLogger("werkzeug")  # Flask's default logger
flask_logger.setLevel(logging.INFO)

for handler in flask_logger.handlers:
    flask_logger.removeHandler(handler)

flask_handler = logging.StreamHandler(stream=log_file)
flask_handler.addFilter(ExcludeAPILoggingFilter())
flask_logger.addHandler(flask_handler)

VAIS_CONSOLE_URL = os.environ.get("VAIS_CONSOLE_URL", "https://console.vais.app")
MODEL_HF_URL = {
    "Phi3V": "microsoft/Phi-3-vision-128k-instruct",
    "Phi3.5V": "microsoft/Phi-3.5-vision-instruct",
    "Qwen2.5VL": "unsloth/Qwen2.5-VL-7B-Instruct",
    "Qwen2VL": "unsloth/Qwen2-VL-7B-Instruct",
    "Qwen2VL-Mini": "unsloth/Qwen2-VL-2B-Instruct",
    "Pixtral": "unsloth/Pixtral-12B-2409-bnb-4bit",
    "Llava1.6-Mistral": "unsloth/llava-v1.6-mistral-7b-hf",
    "Llava1.5": "unsloth/llava-v1.6-mistral-7b-hf",
    "Llama3.2V": "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "BLIP2": "Salesforce/blip2-opt-2.7b"
}

MODEL_HF_URL_LLM = {
    # No unsloth
    "Qwen2.5-7B-Ori": "Qwen/Qwen2.5-7B-Instruct",
    "GPT-4o": "gpt/GPT-4o",
    
    # Update 02-04-2025
    "Gemma-3-12B": "unsloth/gemma-3-12b-it",
    
    # Base Models
    "Phi-3.5-mini": "unsloth/Phi-3.5-mini-instruct",
    "Qwen2.5-32B": "unsloth/Qwen2.5-32B-Instruct",
    "Qwen2.5-7B": "unsloth/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B": "unsloth/Qwen2.5-3B-Instruct",
    "Qwen2.5-1.5B": "unsloth/Qwen2.5-1.5B-Instruct",
    "DeepSeek-R1-Qwen-7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Qwen-1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Llama-8B": "unsloth/DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-32B": "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit",
    "Phi-4": "unsloth/phi-4",
    "Meta-Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "Llama-3.2-3B": "unsloth/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B-Instruct",
    "Llama-3.1-Tulu-3-8B": "unsloth/Llama-3.1-Tulu-3-8B",
    "Llama-3.1-Storm-8B": "unsloth/Llama-3.1-Storm-8B",
    "Gemma-2-9B": "unsloth/gemma-2-9b-bnb-4bit",
    "Gemma-2-2B": "unsloth/gemma-2-2b",
    "SmolLM2-1.7B": "unsloth/SmolLM2-1.7B-Instruct",
    "SmolLM2-360M": "unsloth/SmolLM2-360M-Instruct",
    "SmolLM2-135M": "unsloth/SmolLM2-135M-Instruct",
    "Mistral-7B-Instruct-v0.3": "unsloth/mistral-7b-instruct-v0.3",
    "Mistral-7B": "unsloth/mistral-7b",
    "TinyLlama-Chat": "unsloth/tinyllama-chat",
    "TinyLlama": "unsloth/tinyllama",
    "Phi-3-mini-4k-instruct": "unsloth/Phi-3-mini-4k-instruct",
    "Yi-6B": "unsloth/yi-6b",
    "OpenHermes-2.5-Mistral-7B": "unsloth/OpenHermes-2.5-Mistral-7B",
    "Starling-LM-7B-beta": "unsloth/Starling-LM-7B-beta",
    
    # Coder
    "Qwen2.5-Coder-7B-Instruct":"unsloth/Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-Coder-7B":"unsloth/Qwen2.5-Coder-7B",
    "Qwen2.5-Coder-1.5B-Instruct":"unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen2.5-Coder-1.5B":"unsloth/Qwen2.5-Coder-1.5B",
    "CodeLlama-7B": "unsloth/codellama-7b-bnb-4bit",
    "CodeGemma-7B-IT": "unsloth/codegemma-7b-it",
    
    # Math
    "Qwen2.5-Math-7B-Instruct": "unsloth/Qwen2.5-Math-7B-Instruct",
    "Qwen2.5-Math-7B": "unsloth/Qwen2.5-Math-7B",
    "Qwen2.5-Math-1.5B-Instruct": "unsloth/Qwen2.5-Math-1.5B-Instruct",
    "Qwen2.5-Math-1.5B": "unsloth/Qwen2.5-Math-1.5B",
}

## AGENT Attributes
SYSTEM_MESSAGE = """You are a helpful AI assistant. Please be clear and concise."""

# RAGS
RAGS_BASE_DIR = "src/rags"
os.makedirs(os.path.join(RAGS_BASE_DIR, "pdf"), exist_ok=True)
os.makedirs(os.path.join(RAGS_BASE_DIR, "csv"), exist_ok=True)
os.makedirs(os.path.join(RAGS_BASE_DIR, "image_caption"), exist_ok=True)
os.makedirs(os.path.join(RAGS_BASE_DIR, "pdf_ocr"), exist_ok=True)
os.makedirs(os.path.join(RAGS_BASE_DIR, "image_tabular"), exist_ok=True)
os.makedirs(os.path.join(RAGS_BASE_DIR, "pptx"), exist_ok=True)
os.makedirs(os.path.join(RAGS_BASE_DIR, "image_desc"), exist_ok=True)

@app.route('/run_model', methods=['POST'])
def run_model():
    global is_running, finetune_thread

    if is_running:
        return jsonify({"error": "Finetuning is already in progress. Please wait until it finishes."}), 400

    try:
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        # Retrieve and parse metadata
        metadata_str = request.form.get('data', '')
        if not metadata_str:
            return jsonify({"error": "No metadata provided."}), 400

        try:
            metadata = json.loads(metadata_str)
            print("Received metadata:", metadata)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Could not parse JSON metadata: {e}"}), 400

        data_entries = metadata.get("data", [])
        uploaded_files = request.files.getlist('files')

        if len(uploaded_files) != len(data_entries):
            return jsonify({
                "error": "Number of uploaded files does not match number of data entries."
            }), 400

        saved_files = []
        for idx, file_storage in enumerate(uploaded_files):
            if file_storage and file_storage.filename:
                unique_filename = f"upload_{int(time.time())}_{idx}_{file_storage.filename}"
                save_path = os.path.join(upload_dir, unique_filename)
                file_storage.save(save_path)
                print(f"Uploaded file {idx}: Saved to {save_path}")
                saved_files.append(save_path)
            else:
                return jsonify({"error": f"File at index {idx} is invalid."}), 400

        reconstructed_data = []
        for idx, entry in enumerate(data_entries):
            input_text = entry.get("input", "").strip()
            output_text = entry.get("output", "").strip()

            if not input_text or not output_text:
                return jsonify({
                    "error": f"Data entry at index {idx} is missing 'input' or 'output'."
                }), 400

            image_path = saved_files[idx]  # Map the saved file to the data entry

            reconstructed_data.append({
                "image": image_path,  # Absolute or relative path to the image
                "input": input_text,
                "output": output_text
            })

        model_type = metadata.get("model_type", "Qwen2VL")
        finetune_params = {
            "epochs": metadata.get("epochs", 10),
            "learning_rate": metadata.get("learning_rate", 5e-5),
            "warmup_ratio": metadata.get("warmup_ratio", 0.1),
            "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", 8),
            "optim": metadata.get("optimizer", "adamw_torch"),
            "model_type": MODEL_HF_URL[model_type],
            "peft_r": metadata.get("peft_r", 8),
            "peft_alpha": metadata.get("peft_alpha", 16),
            "peft_dropout": metadata.get("peft_dropout", 0.51),
        }

        def finetune_task(data: List[dict], params: dict):
            global is_running
            is_running = True
            try:
                if model_type in ["Phi3V", "Phi3.5V"]:
                    finetuner = FinetunePhi3V(
                        data=data,
                        epochs=params["epochs"],
                        learning_rate=params["learning_rate"],
                        warmup_ratio=params["warmup_ratio"],
                        gradient_accumulation_steps=params["gradient_accumulation_steps"],
                        optim=params["optim"],
                        model_id=params["model_type"],
                        peft_r=params["peft_r"],
                        peft_alpha=params["peft_alpha"],
                        peft_dropout=params["peft_dropout"],
                    )
                elif model_type in ["BLIP2"]:
                    finetuner = FinetuneBLIP2(
                        data=data,
                        epochs=params["epochs"],
                        learning_rate=params["learning_rate"],
                        warmup_ratio=params["warmup_ratio"],
                        gradient_accumulation_steps=params["gradient_accumulation_steps"],
                        optim=params["optim"],
                        model_id=params["model_type"],
                        peft_r=params["peft_r"],
                        peft_alpha=params["peft_alpha"],
                        peft_dropout=params["peft_dropout"],
                    )
                else:
                    finetuner = FinetuneQwenVL(
                        data=data,
                        epochs=params["epochs"],
                        learning_rate=params["learning_rate"],
                        warmup_ratio=params["warmup_ratio"],
                        gradient_accumulation_steps=params["gradient_accumulation_steps"],
                        optim=params["optim"],
                        model_id=params["model_type"],
                        peft_r=params["peft_r"],
                        peft_alpha=params["peft_alpha"],
                        peft_dropout=params["peft_dropout"],
                    )
                finetuner.run()

                print("Finetuning completed successfully.")

                model_pod_id = metadata.get("model_id")
                response = requests.get(
                    f"{VAIS_CONSOLE_URL}/api/update_status",
                    params={"model_id": model_pod_id, "status": "finished", "is_llm": False}
                )
                if response.status_code == 200:
                    print(f"Successfully notified API about model completion: {model_pod_id}")
                else:
                    print(f"Failed to notify API. Status code: {response.status_code}, Response: {response.text}")

            except Exception as e:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"ERROR during finetuning: {str(e)}\n")
                print(f"ERROR during finetuning: {e}")
                model_id = metadata.get("model_id")
                response = requests.get(
                    f"{VAIS_CONSOLE_URL}/api/update_status",
                    params={"model_id": model_id, "status": "failed", "is_llm": False}
                )
                if response.status_code == 200:
                    print(f"Successfully notified API about model failure: {model_id}")
                else:
                    print(f"Failed to notify API. Status code: {response.status_code}, Response: {response.text}")
                
            finally:
                del finetuner
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                is_running = False

        finetune_thread = threading.Thread(target=finetune_task, args=(reconstructed_data, finetune_params))
        finetune_thread.start()


        return jsonify({
            "message": "Finetuning has been started.",
            "metadata": metadata,
            "saved_files": saved_files
        }), 200

    except Exception as e:
        print("Error in /run_model POST:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/run_model_llm', methods=['POST'])
def run_model_llm():
    global is_running, finetune_thread

    if is_running:
        return jsonify({"error": "Finetuning is already in progress. Please wait until it finishes."}), 400

    try:
        metadata = request.get_json()
        if not metadata:
            return jsonify({"error": "No metadata provided."}), 400

        reconstructed_data = metadata.get("data", [])
        retrain_flag = metadata.get("retrain", None)
        agent_flag = metadata.get("is_agent", None)

        model_type = metadata.get("model_type", "Phi-3.5-mini")
        finetune_params = {
            "epochs": metadata.get("epochs", 10),
            "learning_rate": metadata.get("learning_rate", 5e-5),
            "warmup_ratio": metadata.get("warmup_ratio", 0.1),
            "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", 8),
            "optim": metadata.get("optimizer", "adamw_torch"),
            "model_type": MODEL_HF_URL_LLM[model_type],
            "peft_r": metadata.get("peft_r", 8),
            "peft_alpha": metadata.get("peft_alpha", 16),
            "peft_dropout": metadata.get("peft_dropout", 0.51),
            "retrain_flag": retrain_flag
        }
        model_pod_id = metadata.get("model_id")
        
        try:
            requests.get(
                f"{VAIS_CONSOLE_URL}/api/update_status",
                params={"model_id": model_pod_id, "status": "running", "is_llm": True}
            )
        except:
            pass

        def finetune_task(data: List[dict], params: dict):
            global is_running
            global SYSTEM_MESSAGE
            is_running = True
            try:
                if agent_flag:
                    SYSTEM_MESSAGE = metadata.get("system_prompt", "")
                    if params["model_type"].split("/")[0] != "unsloth":
                        pass
                    else:
                        finetuner = FinetuneLMAgent(
                            data=data,
                            epochs=params["epochs"],
                            learning_rate=params["learning_rate"],
                            warmup_ratio=params["warmup_ratio"],
                            gradient_accumulation_steps=params["gradient_accumulation_steps"],
                            optim=params["optim"],
                            model_id=params["model_type"],
                            peft_r=params["peft_r"],
                            peft_alpha=params["peft_alpha"],
                            peft_dropout=params["peft_dropout"],
                            system_prompt=SYSTEM_MESSAGE,
                        )
                        finetuner.run()
                else:
                    finetuner = FinetuneLM(
                        data=data,
                        epochs=params["epochs"],
                        learning_rate=params["learning_rate"],
                        warmup_ratio=params["warmup_ratio"],
                        gradient_accumulation_steps=params["gradient_accumulation_steps"],
                        optim=params["optim"],
                        model_id=params["model_type"],
                        peft_r=params["peft_r"],
                        peft_alpha=params["peft_alpha"],
                        peft_dropout=params["peft_dropout"],
                        retrain_flag=params["retrain_flag"],
                        system_prompt=SYSTEM_MESSAGE,
                    )
                    finetuner.run()
    
                if agent_flag:
                    if params["model_type"].split("/")[0] != "unsloth":
                        _,_ = initialize_model_no_unsloth(
                            model_id=params["model_type"],
                            checkpoint_root="./model_cp",
                            separator=metadata.get("separator", " "),
                            chunk_size=metadata.get("chunk_size", 4096), 
                            chunk_overlap=metadata.get("chunk_overlap", 50),
                            replace_spaces=metadata.get("replace_spaces", False),
                            delete_urls=metadata.get("delete_urls", False),
                            ocr_model=metadata.get("ocr_model", False),
                            gpt_api_key=metadata.get("gpt_api_key", None),
                            openai_api_key=metadata.get("openai_api_key", None),
                        )
                    else:
                        _,_,_ = initialize_model(
                            model_id=params["model_type"],
                            checkpoint_root="./model_cp",
                            separator=metadata.get("separator", " "),
                            chunk_size=metadata.get("chunk_size", 4096), 
                            chunk_overlap=metadata.get("chunk_overlap", 50),
                            replace_spaces=metadata.get("replace_spaces", False),
                            delete_urls=metadata.get("delete_urls", False),
                            ocr_model=metadata.get("ocr_model", False),
                            gpt_api_key=metadata.get("gpt_api_key", None),
                        )
                print("Optimizing model with olive in background..")
                print("Finetuning completed successfully.")

                response = requests.get(
                    f"{VAIS_CONSOLE_URL}/api/update_status",
                    params={"model_id": model_pod_id, "status": "finished", "is_llm": True}
                )
                if response.status_code == 200:
                    print(f"Successfully notified API about model completion: {model_pod_id}")
                else:
                    print(f"Failed to notify API. Status code: {response.status_code}, Response: {response.text}")

            except Exception as e:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"ERROR during finetuning: {str(e)}\n")
                print(f"ERROR during finetuning: {e}")
                model_id = metadata.get("model_id")
                response = requests.get(
                    f"{VAIS_CONSOLE_URL}/api/update_status",
                    params={"model_id": model_id, "status": "failed", "is_llm": True}
                )
                if response.status_code == 200:
                    print(f"Successfully notified API about model failure: {model_id}")
                else:
                    print(f"Failed to notify API. Status code: {response.status_code}, Response: {response.text}")
                
            finally:  
                if params["model_type"].split("/")[0] != "unsloth":
                    pass
                else:   
                    del finetuner
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                is_running = False
                print("Olive auto-opt started")
                print("Optimizing model...")
                print("This is running in a background process and can be closed.")
                # olive_opt()

        finetune_thread = threading.Thread(target=finetune_task, args=(reconstructed_data, finetune_params))
        finetune_thread.start()
                
        return jsonify({
            "message": "Finetuning has been started.",
            "metadata": metadata,
        }), 200

    except Exception as e:
        print("Error in /run_model_llm POST:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    global log_file_path
    try:
        with open(log_file_path, "w", encoding="utf-8"):
            pass
        return jsonify({"message": "Log file has been cleared."}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to clear logs: {str(e)}"}), 500


@app.route('/logs', methods=['GET'])
def stream_logs():
    global is_running

    if not is_running and not os.path.exists(log_file_path):
        return jsonify({"error": "No model is currently running and no logs found."}), 400

    def log_generator():
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            log_file.seek(0, os.SEEK_END)  # Start at the end of the file

            while is_running:
                line = log_file.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                else:
                    time.sleep(0.1)  # Wait for new lines

            log_file.seek(0)
            for line in log_file:
                yield f"{line.strip()}\n"

    return Response(log_generator(), content_type="text/event-stream")


@app.route('/current_logs', methods=['GET'])
def current_logs():
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {"Content-Type": "text/plain"}
    else:
        return jsonify({"error": "Log file not found"}), 404


@app.route('/download_logs', methods=['GET'])
def download_logs():
    global log_file_path
    if os.path.exists(log_file_path):
        return send_file(log_file_path, as_attachment=True, attachment_filename='model_logs.txt')
    else:
        return jsonify({"error": "Log file not found."}), 404


@app.route('/logs_history', methods=['GET'])
def logs_history():
    try:
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            logs = log_file.read()
        # Wrap logs in <pre> for proper formatting in HTML
        return f"<html><body><pre>{logs}</pre></body></html>", 200
    except FileNotFoundError:
        return "<html><body><h1>Log file not found.</h1></body></html>", 400


@app.route('/stop_model', methods=['POST'])
def stop_model():
    global finetune_thread, is_running

    if not is_running:
        return jsonify({"error": "No model is currently running."}), 400

    try:
        return jsonify({"error": "Stopping the finetuning process is not implemented."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/inference', methods=['POST'])
def inference():
    if 'input' not in request.form:
        return jsonify({"error": "Missing 'input' in form data."}), 400
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file in form data."}), 400
    if 'model_type' not in request.form:
        return jsonify({"error": "Missing 'model_type' in form data."}), 400

    user_input = request.form['input'].strip()
    temperature = float(request.form.get('temperature', 1.0))  # Default: 0.5
    max_tokens = int(request.form.get('max_tokens', 500))      # Default: 500
    model_type = request.form['model_type']

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for 'image'."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        model_id = MODEL_HF_URL[model_type]

        if model_type in ["Phi3V", "Phi3.5V", "BLIP2"]:
            result = run_inference_phi3v(image, user_input, temperature, max_tokens, model_id)
        else:
            result = run_inference_qwenvl(image, user_input, temperature, max_tokens, model_id)

        return jsonify({"result": result}), 200

    except Exception as e:
        print(f"Error in /inference: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/inference_b64', methods=['POST'])
def inference_b64():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON or empty request body."}), 400

    user_input = data.get('input', '').strip()
    image_b64 = data.get('image', '')
    temperature = float(request.form.get('temperature', 0.5)) 
    max_tokens = int(request.form.get('max_tokens', 500))
    model_type = data.get('model_type', '')

    if not user_input:
        return jsonify({"error": "Missing 'input' in JSON."}), 400
    if not image_b64:
        return jsonify({"error": "Missing 'image' in JSON."}), 400
    if not model_type:
        return jsonify({"error": "Missing 'model_type' in JSON."}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        model_id = MODEL_HF_URL[model_type]
        if model_type in ["Phi3V", "Phi3.5V", "BLIP2"]:
            result = run_inference_phi3v(image, user_input, temperature, max_tokens, model_id)
        else:
            result = run_inference_qwenvl(image, user_input, temperature, max_tokens, model_id)
        return jsonify({"result": result}), 200

    except Exception as e:
        print(f"Error in /inference_b64: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.post("/create-session")
def create_session():
    sid = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(Conversation(session_id=sid, history=[]))
        db.commit()
    return {"session_id": sid}


@app.route('/inference-llm', methods=['POST'])
def inference_llm():
    global SYSTEM_MESSAGE
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload."}), 400

    if 'input' not in data:
        return jsonify({"error": "Missing 'input' parameter in JSON."}), 400
    if 'model_type' not in data:
        return jsonify({"error": "Missing 'model_type' parameter in JSON."}), 400

    user_input = data.get("input", "").strip()
    temperature = float(data.get("temperature", 0.5))
    max_tokens = int(data.get("max_tokens", 500))
    model_type = data.get("model_type")
    is_agent = data.get("is_agent", False)
    session_id = data.get("session_id", DEFAULT_SESSION_ID)
    
    conversation_history = load_history(session_id)

    if model_type not in MODEL_HF_URL_LLM:
        return jsonify({"error": f"Unsupported model_type: {model_type}"}), 400
    model_id = MODEL_HF_URL_LLM[model_type]

    try:
        if is_agent:
            if model_id.split("/")[0] != "unsloth":
                result, updated_conversation_history = run_inference_lm_memory_no_unsloth(
                    model_id,
                    user_input,
                    conversation_history=conversation_history,
                    system_prompt=SYSTEM_MESSAGE,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                result, updated_conversation_history = run_inference_lm_memory(
                    model_id,
                    user_input,
                    conversation_history=conversation_history,
                    system_prompt=SYSTEM_MESSAGE,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if updated_conversation_history:
                conversation_history = updated_conversation_history
                save_history(session_id, conversation_history)
        else:
            result = run_inference_lm(user_input, temperature, max_tokens, model_id)
        

        think_match = re.search(r"^(.*?)</think>", result, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            final_result = result[think_match.end():].strip()  # Remove the <think> section
        else:
            think_content = None
            final_result = result.strip()

        response = {"result": final_result}
        if think_content:
            response["think"] = think_content
            
        return jsonify(response), 200

    except Exception as e:
        print(f"Error in /inference: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/inference-llm/stream', methods=['POST'])
def inference_llm_stream():
    global SYSTEM_MESSAGE
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload."}), 400

    if 'input' not in data:
        return jsonify({"error": "Missing 'input' parameter in JSON."}), 400
    if 'model_type' not in data:
        return jsonify({"error": "Missing 'model_type' parameter in JSON."}), 400

    user_input = data.get("input", "").strip()
    temperature = float(data.get("temperature", 0.5))
    max_tokens = int(data.get("max_tokens", 500))
    model_type = data.get("model_type")
    is_agent = data.get("is_agent", False)
    session_id = data.get("session_id", DEFAULT_SESSION_ID)
    
    conversation_history = load_history(session_id)
     
    if model_type not in MODEL_HF_URL_LLM:
        return jsonify({"error": f"Unsupported model_type: {model_type}"}), 400

    model_id = MODEL_HF_URL_LLM[model_type]

    def sse_generator():
        final_response = ""
        think_content = None
        try:
            # choose the appropriate streaming generator
            if is_agent:
                if model_id.split("/")[0] != "unsloth":
                    stream_gen = run_inference_lm_streaming_memory_no_unsloth(
                        model_id=model_id,
                        user_input=user_input,
                        conversation_history=conversation_history,
                        system_prompt=SYSTEM_MESSAGE,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    stream_gen = run_inference_lm_streaming_memory(
                        model_id=model_id,
                        user_input=user_input,
                        conversation_history=conversation_history,
                        system_prompt=SYSTEM_MESSAGE,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            else:
                stream_gen = run_inference_lm_streaming(
                    user_input=user_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_id=model_id
                )

            # stream out tokens (or sub‑chunks)
            for token_chunk in stream_gen:
                final_response += token_chunk

                if len(token_chunk) > 50:
                    # split large chunks into 50‑char pieces
                    for i in range(0, len(token_chunk), 50):
                        sub = token_chunk[i : i + 50]
                        yield f"data: {sub}\n\n"
                        # tiny pause so the client can render incrementally
                        time.sleep(0.01)
                else:
                    yield f"data: {token_chunk}\n\n"

            # emit think event if present
            match = re.search(r"^(.*?)</think>", final_response, re.DOTALL)
            if match:
                think_content = match.group(1).strip()
                final_response = final_response[match.end():].strip()
                yield f"event: think\ndata: {think_content}\n\n"

            # final end event
            # yield "event: end\ndata: [END]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            
        finally:
            save_history(session_id, conversation_history)

    return Response(sse_generator(), mimetype='text/event-stream')


video_inference_thread = None
is_video_inferencing = False
video_inference_logs_path = "video_inference_logs.txt"

@app.route('/inference-video', methods=['POST'])
def inference_video():
    global video_inference_thread, is_video_inferencing

    if 'input' not in request.form:
        return jsonify({"error": "Missing 'input' in form data."}), 400

    user_input = request.form['input'].strip()
    temperature = float(request.form.get('temperature', 1.0))
    max_tokens = int(request.form.get('max_tokens', 500))
    fps = float(request.form.get('fps', 1.0))  # Default: 0.5

    if 'model_type' not in request.form:
        return jsonify({"error": "Missing 'model_type' in form data."}), 400
    model_type = request.form['model_type']

    if 'video' not in request.files:
        return jsonify({"error": "Missing 'video' file in form data."}), 400
    file = request.files['video']

    saved_video_path = os.path.join("uploads", f"{int(time.time())}_{file.filename}")
    os.makedirs("uploads", exist_ok=True)
    file.save(saved_video_path)

    if is_video_inferencing:
        return jsonify({"error": "A video inference job is already running."}), 400

    model_id = MODEL_HF_URL[model_type]

    try:
        open(video_inference_logs_path, "w").close()
    except:
        pass

    def background_video_inference():
        global is_video_inferencing
        is_video_inferencing = True
        try:
            result = run_inference_qwenvl_video(
                video_path=saved_video_path,
                user_input=user_input,
                temperature=temperature,
                max_tokens=max_tokens,
                fps=fps,
                model_id=model_id
            )
            with open(video_inference_logs_path, "a", encoding="utf-8") as f:
                f.write(result + "\n")

        except Exception as e:
            with open(video_inference_logs_path, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {str(e)}\n")
        finally:
            is_video_inferencing = False

    video_inference_thread = threading.Thread(target=background_video_inference)
    video_inference_thread.start()

    return jsonify({
        "message": "Video inference has started in the background."
    }), 200


@app.route('/video_inference_logs', methods=['GET'])
def get_video_inference_logs():
    if os.path.exists(video_inference_logs_path):
        with open(video_inference_logs_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content, mimetype='text/plain')
    else:
        return jsonify({"error": "No video inference log file found."}), 404
    
# RAGS
@app.route('/upload_file', methods=['POST'])
def upload_file():
    file_type = request.form.get("type")
    if 'file' in request.files:
        f = request.files['file']
        if f.filename == '':
            return jsonify({"error": "No selected file"}), 400
        filename = secure_filename(f.filename)
        subdir = None
        if file_type in ["pdf", "txt"]:
            subdir = "pdf"
        elif file_type == "csv":
            subdir = "csv"
        elif file_type == "chartImage":
            subdir = "image_tabular"
        elif file_type == "imageOcr":
            subdir = "image_caption"
        elif file_type == "pptx":
            subdir = "pptx"
        elif file_type == "imageDesc":
            subdir = "image_desc"
        elif file_type == "pdfOcr":
            subdir = "pdf_ocr"
        else:
            subdir = "pdf"

        save_path = os.path.join(RAGS_BASE_DIR, subdir, filename)
        f.save(save_path)

        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "type": file_type,
            "path": save_path
        }), 200

    website_url = request.form.get("website_url")
    if website_url:
        # Append to website.txt
        txt_path = os.path.join(RAGS_BASE_DIR, "website.txt")
        with open(txt_path, "a", encoding="utf-8") as w:
            w.write(website_url.strip() + "\n")

        return jsonify({
            "message": "Website URL saved successfully",
            "url": website_url
        }), 200

    return jsonify({"error": "No file or website URL provided"}), 400


@app.route('/delete_rag_item', methods=['POST'])
def delete_rag_item():
    data = request.get_json() or {}
    item_type = data.get("type")
    name = data.get("name")

    if not item_type or not name:
        return jsonify({"error": "Missing type or name"}), 400

    if item_type == "url":
        txt_path = os.path.join(RAGS_BASE_DIR, "website.txt")
        if not os.path.exists(txt_path):
            return jsonify({"error": "website.txt not found"}), 404

        updated_lines = []
        removed_any = False
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == name.strip():
                    removed_any = True
                    continue
                updated_lines.append(line)

        if removed_any:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)
            return jsonify({"message": f"URL '{name}' deleted."}), 200
        else:
            return jsonify({"error": f"URL '{name}' not found in file."}), 404
    else:
        if item_type in ["pdf", "txt"]:
            subdir = "pdf"
        elif item_type == "csv":
            subdir = "csv"
        elif item_type == "chartImage":
            subdir = "image_tabular"
        elif item_type == "imageOcr":
            subdir = "image_caption"
        elif item_type == "pptx":
            subdir = "pptx"
        elif item_type == "imageDesc":
            subdir = "image_desc"
        elif item_type == "pdfOcr":
            subdir = "pdf_ocr"
        else:
            subdir = "pdf"

        file_path = os.path.join(RAGS_BASE_DIR, subdir, name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"message": f"File '{name}' deleted from {subdir}."}), 200
        else:
            return jsonify({"error": f"File '{name}' not found on server."}), 404


if __name__ == "__main__":
    app.run(host='0.5.0.5', port=5000, debug=True)
