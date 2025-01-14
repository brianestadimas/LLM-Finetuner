import subprocess
import threading
import time
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

is_running = False
process = None 
log_file_path = "model_logs.txt" 


def run_model_in_background(epoch):
    global is_running, process

    is_running = True

    # Open the log file in write mode and pass it to subprocess
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        try:
            # Start the subprocess
            cmd = ["python", "src/phi3v.py", f"--epoch={epoch}"]
            process = subprocess.Popen(
                cmd,
                stdout=log_file,  # Redirect stdout to the log file
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout (merged in the log file)
                text=True,
                bufsize=1
            )
            process.wait()  # Wait for the subprocess to finish
        except Exception as e:
            # Log any exception to the log file
            log_file.write(f"ERROR: {str(e)}\n")
        finally:
            is_running = False  # Mark process as no longer running


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Finetune VLM"}), 200


@app.route('/run_model', methods=['GET'])
def run_model():
    """
    Starts the model in a background thread, logging all output to a file.
    """
    global is_running

    if is_running:
        return jsonify({"error": "Model is already running. Wait for it to finish."}), 400

    epoch = int(request.args.get("epoch", 100))

    # Clear the log file before starting a new run
    open(log_file_path, "w").close()

    # Start the model in a background thread
    thread = threading.Thread(target=run_model_in_background, args=(epoch,))
    thread.start()

    return jsonify({"message": f"Model started with epoch {epoch}. Connect to /logs or /logs_history for output."}), 200


@app.route('/logs', methods=['GET'])
def stream_logs():
    """
    Stream logs from the file in real-time using Server-Sent Events (SSE).
    """
    global is_running

    if not is_running:
        return jsonify({"error": "No model is currently running."}), 400

    def log_generator():
        """
        Generator that yields lines from the log file in real-time.
        """
        with open(log_file_path, "r", encoding="utf-8") as log_file:
            log_file.seek(0, 2)  # Move to the end of the file

            while is_running:
                line = log_file.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                else:
                    time.sleep(0.1)  # Sleep briefly to avoid busy waiting

    return Response(log_generator(), content_type="text/event-stream")


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
    global process, is_running

    if not is_running:
        return jsonify({"error": "No model is currently running."}), 400

    try:
        if process and process.poll() is None:  # Check if the process is still running
            process.terminate()  # Terminate the process
            process.wait()  # Wait for the process to terminate
        is_running = False
        return jsonify({"message": "Model stopped successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
