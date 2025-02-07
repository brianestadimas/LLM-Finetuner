import os, json
from datetime import datetime
from flask import Flask, request, jsonify
import requests
from src import db, Run, create_app
from src.config import Config
import runpod
from flask_cors import CORS
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from src import User, Run

app = create_app()
CORS(app)

runpod.api_key = Config.RUNPOD_KEY

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')

@app.route('/finetune', methods=['POST'])
def finetune_route():
    """
    Initiate a finetuning run for a user with additional parameters.
    """
    data = request.get_json()

    # Validate required parameters
    email = data.get('email')
    model_name = data.get('model_name')
    model_type = data.get('model_type')
    is_llm = data.get('is_llm', False)
    description = data.get('description')
    runpod_api_key = data.get('runpod_api_key')

    if not email or not model_name or not model_type or not description:
        return jsonify({"error": "email, model_name, model_type, and description are required"}), 400

    # Retrieve user by email
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check if the user already has an active run
    if user.run_id is not None:
        active_run = Run.query.get(user.run_id)
        if active_run and active_run.status in ["pending", "running"]:
            return jsonify({"error": "User already has an active simulation"}), 400

    # Create a new run
    new_run = Run(
        user_id=user.id,
        status="pending",
        model_name=model_name,
        model_type=model_type,
        is_llm=is_llm,
        description=description
    )
    db.session.add(new_run)
    db.session.commit()

    run_id = new_run.id

    # Define GPU fallback list
    gpu_types = [
        # "NVIDIA GeForce RTX 3070",
        # "NVIDIA GeForce RTX 3080",
        # 
        "NVIDIA A40",
        "NVIDIA GeForce RTX 3090",
    ]

    podcast_id = None
    last_error = None
    
    if runpod_api_key:
        runpod.api_key = runpod_api_key

    # Attempt pod creation with fallback GPUs
    for gpu in gpu_types:
        try:
            if model_type in ["Phi3V", "Phi3.5V", "Qwen2VL", "Qwen2VL-Mini"]:
                container_size = 20
            elif model_type in ["DeepSeek-R1-Distill-32B"]:
                container_size = 60
            else:
                container_size = 36
            pod = runpod.create_pod(
                name=f"FT-Run-{run_id}",
                image_name="brianarfeto/finetune-vlm:latest",
                gpu_type_id=gpu,
                gpu_count=1,
                volume_in_gb=10,
                container_disk_in_gb=container_size,
                ports="5000/http",
            )
            podcast_id = pod.get('id') + "-5000"
            if podcast_id:
                break  # If successful, exit loop

        except Exception as e:
            last_error = str(e)  # Store the error and try the next GPU

    if not podcast_id:
        runpod.api_key = Config.RUNPOD_KEY
        db.session.rollback()
        try:
            db.session.delete(new_run)
            user.run_id = None
            db.session.commit()
        except Exception as inner_error:
            return jsonify({"error": f"Failed to clean up after error: {str(inner_error)}"}), 500

        return jsonify({"error": f"Failed to create a pod: {last_error}"}), 500

    # Update the run and associate it with the user
    new_run.status = "running"
    new_run.podcast_id = podcast_id
    user.run_id = run_id
    db.session.commit()
    
    runpod.api_key = Config.RUNPOD_KEY

    return jsonify({
        "message": "Finetuning initiated successfully.",
        "run_id": run_id,
        "podcast_id": podcast_id,
        "model_name": new_run.model_name,
        "model_type": new_run.model_type,
        "description": new_run.description,
        "status": new_run.status,
    }), 200

@app.route('/finished_finetuning', methods=['GET'])
def finished_finetuning():
    """
    Removes the run_id from the associated user and marks the run as completed or removed.
    Accepts podcast_id as a parameter.
    """
    params = request.args
    podcast_id = params.get('podcast_id')
    is_llm = params.get('is_llm', False)

    if not podcast_id:
        return jsonify({"error": "Podcast ID is required"}), 400

    # Find the run associated with the podcast_id
    run = Run.query.filter_by(podcast_id=podcast_id).first()
    if not run:
        return jsonify({"error": "Run with the provided podcast ID not found"}), 404

    try:
        # Update the run status to "finished" or "removed"
        run.status = "finished"  # or "removed" based on your logic
        run.is_llm = is_llm

        # Find the user associated with the run_id
        user = User.query.filter_by(run_id=run.id).first()
        if user:
            user.run_id = None

        # Commit changes to the database
        db.session.commit()

        return jsonify({"message": "Finetuning finished successfully", "run_id": run.id}), 200

    except Exception as e:
        db.session.rollback()  # Rollback in case of errors
        return jsonify({"error": f"Failed to finish finetuning: {str(e)}"}), 500

@app.route('/delete', methods=['GET'])
def delete_run():
    """
    Deletes a run based on the provided podcast_id.
    If the associated pod does not exist, it still proceeds to delete the run instance.
    """
    podcast_id = request.args.get('podcast_id')
    if not podcast_id:
        return jsonify({"error": "Podcast ID is required"}), 400

    pod_id = podcast_id.replace("-5000", "")

    run = Run.query.filter_by(podcast_id=podcast_id).first()
    if not run:
        return jsonify({"error": "Run not found"}), 404

    try:
        # Attempt to stop the pod
        runpod.stop_pod(pod_id)
        runpod.terminate_pod(pod_id)

    except Exception as e:
        error_message = str(e)

        # Check if the error message indicates the pod does not exist
        if "Attempted to stop pod that does not exist" in error_message:
            print(f"Pod {pod_id} does not exist, proceeding with run deletion...")
        else:
            return jsonify({"error": f"Failed to delete run: {error_message}"}), 500
    try:
        # Remove the run from the database
        db.session.delete(run)
        db.session.commit()
        return jsonify({"message": f"Run with podcast_id {podcast_id} deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@app.route('/run_list', methods=['GET'])
def run_list():
    user_email = request.args.get('email')
    if not user_email:
        return jsonify({"error": "Email is required"}), 400

    try:
        # Find the user by email
        user = User.query.filter_by(email=user_email).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get the runs associated with the user
        user_runs = Run.query.filter_by(user_id=user.id).order_by(Run.id.desc()).all()

        # Format the runs as a list of dictionaries
        run_data = [
            {
                "run_id": run.id,
                "podcast_id": run.podcast_id,
                "status": run.status,
                "model_name": run.model_name,
                "model_type": run.model_type,
                "description": run.description,
                "is_llm": run.is_llm,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            }
            for run in user_runs
        ]

        return jsonify(run_data), 200

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve run list: {str(e)}"}), 500


@app.route('/get_podcast', methods=['GET'])
def get_podcast():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user.run_id is None:
        return jsonify({
            "message": "User has no active simulation",
            "run_id": "",
            "status": "",
            "podcast_id": "",
        }), 200

    active_run = Run.query.get(user.run_id)
    if not active_run:
        return jsonify({
            "message": "User has no active simulation",
            "run_id": "",
            "status": "",
            "podcast_id": ""
        }), 200

    return jsonify({
        "message": "User has an active simulation",
        "run_id": user.run_id,
        "status": active_run.status,
        "podcast_id": active_run.podcast_id
    }), 200


@app.route("/oauth/callback")
def oauth_callback():
    code = request.args.get("code")
    if not code:
        return "Missing code", 400

    # Exchange authorization code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    r = requests.post(token_url, data=data)
    tokens = r.json()

    return f"Tokens: {tokens}"


@app.route('/api/login/google', methods=['POST'])
def google_login():
    data = request.get_json()
    token = data.get('credential')  # ID token from the frontend
    if not token:
        return jsonify({"error": "No credential token provided"}), 400
    
    try:
        # Verify the ID token using Google's libraries
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID  # Replace with your actual client ID
        )
        # Extract user info from the verified token
        user_email = idinfo.get('email')
        user_name = idinfo.get('name')
        user_picture = idinfo.get('picture')

        # Upsert user into your database
        existing_user = User.query.filter_by(email=user_email).first()
        if not existing_user:
            new_user = User(email=user_email, name=user_name, picture=user_picture)
            db.session.add(new_user)
            db.session.commit()
            user_id = new_user.id
        else:
            existing_user.name = user_name
            existing_user.picture = user_picture
            db.session.commit()
            user_id = existing_user.id

        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user_id,
                "email": user_email,
                "name": user_name,
                "picture": user_picture
            }
        }), 200
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/inference/<model_id>', methods=['POST'])
def inference(model_id):
    model_endpoint = f"https://{model_id}.proxy.runpod.net/inference"

    input_text = request.form.get("input")
    temperature = request.form.get("temperature", 0.0)  # Default: 0.0
    max_tokens = request.form.get("max_tokens", 500)    # Default: 500
    image = request.files.get("image")

    if not input_text or not image:
        return jsonify({"error": "Missing required parameters: input and/or image"}), 400

    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type 
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    # Prepare request payload
    files = {"image": (image.filename, image.stream, image.mimetype)}
    data = {
        "input": input_text,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type
    }
    try:
        response = requests.post(model_endpoint, files=files, data=data)
        return jsonify(response.json()), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


@app.route('/inference_b64/<model_id>', methods=['POST'])
def inference_b64(model_id):
    model_endpoint = f"https://{model_id}.proxy.runpod.net/inference_b64"

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON or empty request body."}), 400

    input_text = data.get("input", "").strip()
    image_b64 = data.get("image", "")
    
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type 
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    temperature = float(data.get("temperature", 0.0))
    max_tokens = int(data.get("max_tokens", 500))

    if not input_text or not image_b64:
        return jsonify({"error": "Missing required parameters: input and/or image"}), 400

    payload = {
        "input": input_text,
        "image": image_b64,  # Forward base64 as a string
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type
    }

    try:
        response = requests.post(model_endpoint, json=payload, timeout=120)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request to model API failed: {str(e)}"}), 500


@app.route('/inference-llm/<model_id>', methods=['POST'])
def inference_llm(model_id):
    # Retrieve JSON payload from the request body.
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    input_text = data.get("input")
    temperature = data.get("temperature", 0.0)  # Default: 0.0
    max_tokens = data.get("max_tokens", 1000)    # Default: 500

    if not input_text or not model_id:
        return jsonify({"error": "Missing required parameters: input and/or model_id"}), 400

    # Build the model endpoint using the provided model_id.
    model_endpoint = f"https://{model_id}.proxy.runpod.net/inference-llm"

    # Look up the model's record.
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Invalid model_id (podcast_id) or model not found."}), 404

    model_type = run.model_type 
    if not model_type:
        return jsonify({"error": "Model type not found for this model_id."}), 400

    # Prepare JSON payload for the inference request.
    payload = {
        "input": input_text,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_type": model_type
    }
    try:
        # Post JSON payload to the model endpoint.
        response = requests.post(model_endpoint, json=payload)
        return jsonify(response.json()), response.status_code

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


@app.route('/update_status', methods=['GET'])
def update_status():
    model_id = request.args.get('model_id')
    status = request.args.get('status')
    is_llm_str = request.args.get('is_llm', False)

    if is_llm_str is not None:
        if is_llm_str.lower() in ['true', '1', 'yes']:
            is_llm = True
        elif is_llm_str.lower() in ['false', '0', 'no']:
            is_llm = False
        else:
            return jsonify({"error": "Invalid value for is_llm. Use true or false."}), 400
    else:
        is_llm = False

    if not model_id or not status:
        return jsonify({"error": "model_id and status are required"}), 400

    # Find the corresponding run
    run = Run.query.filter_by(podcast_id=model_id).first()
    if not run:
        return jsonify({"error": "Run not found for the given model_id"}), 404

    try:
        # Update run status
        if status.lower() not in ["finished", "failed"]:
            return jsonify({"error": "Invalid status. Must be 'finished' or 'failed'"}), 400

        run.status = status.lower()
        run.is_llm = is_llm

        # Find the user associated with this run
        user = User.query.filter_by(run_id=run.id).first()
        if user:
            user.run_id = None  # Nullify user's run_id since the process is completed

        # Commit changes to database
        db.session.commit()
        
        if status.lower() == "failed":
            pod_id = model_id.replace("-5000", "")
            runpod.stop_pod(pod_id)
            runpod.terminate_pod(pod_id)

        return jsonify({
            "message": f"Run {run.id} status updated to '{status}' and user run_id nullified.",
            "run_id": run.id,
            "new_status": run.status
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to update status: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)