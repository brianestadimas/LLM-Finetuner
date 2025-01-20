import io
import threading
import time
from flask import Flask, Response, jsonify, request, send_file
import torch
from src.phi3v import FinetunePhi3V
from flask_cors import CORS
import os
import json
from typing import List
import sys
import logging
import re
import base64
from PIL import Image
from src.inference_phi3v import run_inference

app = Flask(__name__)
CORS(app)

# Global variables to manage the finetuning process
is_running = False
finetune_thread = None

log_file_path = "model_logs.txt"

# Custom logging filter to exclude API access logs
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

# Redirect stdout and stderr to the log file
log_file = open(log_file_path, "a", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

# Configure Flask logging
flask_logger = logging.getLogger("werkzeug")  # Flask's default logger
flask_logger.setLevel(logging.INFO)

# Replace default handlers with filtered handlers
for handler in flask_logger.handlers:
    flask_logger.removeHandler(handler)

flask_handler = logging.StreamHandler(stream=log_file)
flask_handler.addFilter(ExcludeAPILoggingFilter())
flask_logger.addHandler(flask_handler)


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

        # Extract 'data' entries
        data_entries = metadata.get("data", [])
        if not isinstance(data_entries, list) or not data_entries:
            return jsonify({"error": "Invalid or empty 'data' provided in metadata."}), 400

        # Save uploaded files
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

        # Reconstruct data for ImageTextDataset
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

        # Extract fine-tuning parameters from metadata with default values
        finetune_params = {
            "epochs": metadata.get("epochs", 1),
            "learning_rate": metadata.get("learning_rate", 1e-4),
            "warmup_ratio": metadata.get("warmup_ratio", 0.1),
            "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", 64),
            "optim": metadata.get("optim", "adamw_torch"),
            "model_id": metadata.get("model_id", "microsoft/Phi-3-vision-128k-instruct"),
            "peft_r": metadata.get("peft_r", 8),
            "peft_alpha": metadata.get("peft_alpha", 16),
            "peft_dropout": metadata.get("peft_dropout", 0.05),
        }

        # Define the finetuning task
        def finetune_task(data: List[dict], params: dict):
            global is_running
            is_running = True
            try:
                # Initialize the finetuner
                finetuner = FinetunePhi3V(
                    data=data,
                    epochs=params["epochs"],
                    learning_rate=params["learning_rate"],
                    warmup_ratio=params["warmup_ratio"],
                    gradient_accumulation_steps=params["gradient_accumulation_steps"],
                    optim=params["optim"],
                    model_id=params["model_id"],
                    peft_r=params["peft_r"],
                    peft_alpha=params["peft_alpha"],
                    peft_dropout=params["peft_dropout"],
                )

                # Run the finetuning process
                finetuner.run()

                # Optionally, handle post-finetuning steps here
                print("Finetuning completed successfully.")

            except Exception as e:
                # Log any exceptions to the log file
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"ERROR during finetuning: {str(e)}\n")
                print(f"ERROR during finetuning: {e}")
            finally:
                # Clear references to the finetuner and underlying model
                del finetuner
                # Force garbage collection
                import gc
                gc.collect()
                # Clear PyTorch cache on GPU
                torch.cuda.empty_cache()

                is_running = False

        # Start the finetuning in a background thread
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

            # After finetuning completes, stream remaining lines
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
    """
    Endpoint to download the model_logs.txt file.
    """
    global log_file_path
    if os.path.exists(log_file_path):
        return send_file(log_file_path, as_attachment=True, attachment_filename='model_logs.txt')
    else:
        return jsonify({"error": "Log file not found."}), 404


@app.route('/logs_history', methods=['GET'])
def logs_history():
    """
    Returns all logs from the log file rendered as HTML.
    """
    try:
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            logs = log_file.read()
        # Wrap logs in <pre> for proper formatting in HTML
        return f"<html><body><pre>{logs}</pre></body></html>", 200
    except FileNotFoundError:
        return "<html><body><h1>Log file not found.</h1></body></html>", 400


@app.route('/stop_model', methods=['POST'])
def stop_model():
    """
    Stops the currently running model process.
    """
    global finetune_thread, is_running

    if not is_running:
        return jsonify({"error": "No model is currently running."}), 400

    try:
        # Since the finetuning is running in a background thread,
        # implementing a stop mechanism requires modifying FinetunePhi3V to support it.
        # One approach is to use a threading.Event to signal the finetuner to stop.
        # For simplicity, we'll inform the user that stopping is not implemented.

        return jsonify({"error": "Stopping the finetuning process is not implemented."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/inference', methods=['POST'])
def inference():
    """
    Receives multipart/form-data with 'input' (text prompt) and 'image' (file).
    Returns JSON with the model's generated text.
    """
    # Check if we got 'input'
    if 'input' not in request.form:
        return jsonify({"error": "Missing 'input' in form data."}), 400
    user_input = request.form['input'].strip()

    # Check if we got an image
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file in form data."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for 'image'."}), 400

    try:
        # Open as PIL image
        image = Image.open(file.stream).convert("RGB")
        result = run_inference(image, user_input)
        return jsonify({"result": result}), 200
    except Exception as e:
        print(f"Error in /inference: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/inference_b64', methods=['POST'])
def inference_b64():
    """
    Receives JSON: {'input': <text prompt>, 'image': <base64 string>}.
    Returns JSON with the model's generated text.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON or empty request body."}), 400

    user_input = data.get('input', '').strip()
    image_b64 = data.get('image', '')

    if not user_input:
        return jsonify({"error": "Missing 'input' in JSON."}), 400
    if not image_b64:
        return jsonify({"error": "Missing 'image' in JSON."}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        result = run_inference(image, user_input)
        return jsonify({"result": result}), 200
    except Exception as e:
        print(f"Error in /inference_b64: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
