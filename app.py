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

@app.route('/run_model', methods=['POST'])
def finetune_route_post():
    try:
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)  

        metadata_str = request.form.get('data', '')
        if not metadata_str:
            return jsonify({"error": "No metadata provided"}), 400

        try:
            metadata = json.loads(metadata_str)
            print(metadata)
        except Exception as e:
            return jsonify({"error": f"Could not parse JSON: {e}"}), 400

        saved_files = []
        for file_key in request.files:
            file_storage = request.files[file_key]
            if file_storage and file_storage.filename:
                save_path = os.path.join(upload_dir, file_storage.filename)
                file_storage.save(save_path)

                # Log and store the path
                print(f"Uploaded file: {file_key} -> Saved to {save_path}")
                saved_files.append(save_path)

        return jsonify({
            "message": "Finetuning POST data received",
            "metadata": metadata,
            "saved_files": saved_files
        }), 200

    except Exception as e:
        print("Error in /run_model POST:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/finetune', methods=['GET'])
def finetune_route():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Check if the user already has an active run
    if user.run_id is not None:
        active_run = Run.query.get(user.run_id)
        if active_run and active_run.status in ["pending", "running"]:
            return jsonify({"error": "User already has an active simulation"}), 400

    # Create a new run
    new_run = Run(status="pending")
    db.session.add(new_run)
    db.session.commit()

    run_id = new_run.id
    try:
        # Create a new pod and retrieve the podcast_id
        pod = runpod.create_pod(
            name=f"FT-Run-{run_id}",
            image_name="brianarfeto/finetune-vlm:latest",
            gpu_type_id="NVIDIA GeForce RTX 3070",
            gpu_count=1,
            volume_in_gb=10,
            container_disk_in_gb=5,
            ports="5000/http",
        )
        podcast_id = pod.get('id') + "-5000"
        if not podcast_id:
            raise ValueError("Failed to retrieve podcast_id from the pod creation response")

        # Update the run and associate it with the user
        new_run.status = "running"
        new_run.podcast_id = podcast_id
        user.run_id = run_id
        db.session.commit()

        return jsonify({
            "message": "Finetuning initiated successfully.",
            "run_id": run_id,
            "podcast_id": podcast_id
        }), 200

    except Exception as e:
        # Rollback changes if pod creation fails
        db.session.rollback()  # Rollback any uncommitted changes
        try:
            # Delete the run and nullify the user's run_id
            db.session.delete(new_run)
            user.run_id = None
            db.session.commit()
        except Exception as inner_error:
            return jsonify({"error": f"Failed to clean up after error: {str(inner_error)}"}), 500

        return jsonify({"error": str(e)}), 500



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





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)