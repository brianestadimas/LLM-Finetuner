import io
import threading
import time
from flask import Flask, Response, jsonify, request, send_file
import requests
import torch
from src.phi3v import FinetunePhi3V
from src.qwenvl import FinetuneQwenVL
from src.llms import FinetuneLM
from flask_cors import CORS
import os
import json
from typing import List
import sys
import logging
import re
import base64
from PIL import Image
from src.inference_phi3v import run_inference_phi3v
from src.inference_qwenvl import run_inference_qwenvl
from src.inference_llms import run_inference_lm

app = Flask(__name__)
CORS(app)

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


MODEL_HF_URL = {
    "Phi3V": "microsoft/Phi-3-vision-128k-instruct",
    "Phi3.5V": "microsoft/Phi-3.5-vision-instruct",
    "Qwen2VL": "unsloth/Qwen2-VL-7B-Instruct",
    "Qwen2VL-Mini": "unsloth/Qwen2-VL-2B-Instruct",
    "Pixtral": "unsloth/Pixtral-12B-2409-bnb-4bit",
    "Llava1.6-Mistral": "unsloth/llava-v1.6-mistral-7b-hf",
    "Llava1.5": "unsloth/llava-v1.6-mistral-7b-hf",
    "Llama3.2V": "unsloth/Llama-3.2-11B-Vision-bnb-4bit"
}

MODEL_HF_URL_LLM = {
    "Phi-3.5-mini": "unsloth/Phi-3.5-mini-instruct",
    "Qwen2.5-7B": "unsloth/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B": "unsloth/Qwen2.5-3B-Instruct",
    "Qwen2.5-1.5B": "unsloth/Qwen2.5-1.5B-Instruct",
    "DeepSeek-R1-Qwen-7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Qwen-1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Llama-8B": "unsloth/DeepSeek-R1-Distill-Llama-8B",
    "Phi-4": "unsloth/phi-4",
    "Meta-Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
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
        if not isinstance(data_entries, list) or not data_entries:
            return jsonify({"error": "Invalid or empty 'data' provided in metadata."}), 400

        uploaded_files = request.files.getlist('files')
        if len(uploaded_files) != len(data_entries):
            return jsonify({
                "error": "Number of uploaded files does not match number of data entries."
            }), 400

        saved_files = []
        for idx, file_storage in enumerate(uploaded_files):
            if file_storage and file_storage.filename:
                # Ensure unique filenames to prevent overwriting
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
            "peft_dropout": metadata.get("peft_dropout", 0.01),
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
                    "https://console.vais.app/api/update_status",
                    params={"model_id": model_pod_id, "status": "finished"}
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
                    "https://console.vais.app/api/update_status",
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

        data_entries = metadata.get("data", [])
        if not isinstance(data_entries, list) or not data_entries:
            return jsonify({"error": "Invalid or empty 'data' provided in metadata."}), 400

        reconstructed_data = []
        for idx, entry in enumerate(data_entries):
            input_text = entry.get("input", "").strip()
            output_text = entry.get("output", "").strip()

            if not input_text or not output_text:
                return jsonify({
                    "error": f"Data entry at index {idx} is missing 'input' or 'output'."
                }), 400

            reconstructed_data.append({
                "input": input_text,
                "output": output_text
            })

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
            "peft_dropout": metadata.get("peft_dropout", 0.01),
        }

        def finetune_task(data: List[dict], params: dict):
            global is_running
            is_running = True
            try:
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
                )
                finetuner.run()

                print("Finetuning completed successfully.")

                model_pod_id = metadata.get("model_id")
                response = requests.get(
                    "https://console.vais.app/api/update_status",
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
                    "https://console.vais.app/api/update_status",
                    params={"model_id": model_id, "status": "failed"}
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
        }), 200

    except Exception as e:
        print("Error in /run_model_llm POST:", str(e))
        return jsonify({"error": str(e)}), 500



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
    temperature = float(request.form.get('temperature', 0.0))  # Default: 0.0
    max_tokens = int(request.form.get('max_tokens', 500))      # Default: 500
    model_type = request.form['model_type']

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for 'image'."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        model_id = MODEL_HF_URL[model_type]

        if model_type in ["Phi3V", "Phi3.5V"]:
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
    temperature = float(request.form.get('temperature', 0.0)) 
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
        if model_type in ["Phi3V", "Phi3.5V"]:
            result = run_inference_phi3v(image, user_input, temperature, max_tokens, model_id)
        else:
            result = run_inference_qwenvl(image, user_input, temperature, max_tokens, model_id)
        return jsonify({"result": result}), 200

    except Exception as e:
        print(f"Error in /inference_b64: {str(e)}")
        return jsonify({"error": str(e)}), 500

from flask import request, jsonify

@app.route('/inference-llm', methods=['POST'])
def inference_llm():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload."}), 400

    if 'input' not in data:
        return jsonify({"error": "Missing 'input' parameter in JSON."}), 400
    if 'model_type' not in data:
        return jsonify({"error": "Missing 'model_type' parameter in JSON."}), 400

    user_input = data.get("input", "").strip()
    temperature = float(data.get("temperature", 0.0))
    max_tokens = int(data.get("max_tokens", 500))
    model_type = data.get("model_type")

    if model_type not in MODEL_HF_URL_LLM:
        return jsonify({"error": f"Unsupported model_type: {model_type}"}), 400
    model_id = MODEL_HF_URL_LLM[model_type]

    try:
        result = run_inference_lm(user_input, temperature, max_tokens, model_id)
        return jsonify({"result": result}), 200

    except Exception as e:
        print(f"Error in /inference: {str(e)}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
