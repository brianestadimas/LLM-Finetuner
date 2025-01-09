# app.py
from datetime import datetime
from flask import Flask, request, jsonify
from src import db, Run, create_app
from src.ft_phi3v import finetune_phi3v
from src.config import Config
import runpod

app = create_app()

runpod.api_key = Config.RUNPOD_KEY

@app.route('/')
def home():
    print("Welcome to the Finetune VLM")
    return jsonify({"message": "Welcome to the Finetune VLM"}), 200

@app.route('/finetune', methods=['GET'])
def finetune_route():
    params = request.args
    dataset_path = params.get('dataset_path', '')
    epochs = int(params.get('epochs', 1))
    batch_size = int(params.get('batch_size', 1))
    
    # Optional: Log the request or create a record in the database
    new_run = Run(
        status="pending"
    )
    db.session.add(new_run)
    db.session.commit()
    
    run_id = new_run.id 
    pod = runpod.create_pod(
        name=f"Test-Run-{run_id}",
        image_name="huggingface/transformers-pytorch-gpu", 
        gpu_type_id="NVIDIA GeForce RTX 3070",  
        gpu_count=1,
        volume_in_gb=10,
        container_disk_in_gb=5,
        ports="5000/http",  # Expose port 5000 for potential future use
        docker_args=(
            "python /workspace/src/ft_phi3v.py"
        )
    )
    print(pod)
    
    
    try:
        finetune_phi3v(run_id)
        new_run.status = "running"
        db.session.commit()
        
        return jsonify({
            "message": "Finetuning initiated successfully.",
            "run_id": run_id
        }), 200

    except Exception as e:
        # If something fails, log it and return error
        new_run.status = "failed"
        db.session.commit()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)