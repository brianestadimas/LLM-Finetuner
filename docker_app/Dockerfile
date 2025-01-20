# Dockerfile

FROM python:3.11-slim

# Create working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# RUN pip install packaging==23.2
# Install dependencies
RUN pip install -r requirements.txt 

# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=$CUDA_HOME/bin:$PATH
# ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# RUN pip install flash-attn --no-build-isolation

# RUN pip install flash-attn==2.7.2.post1
# Copy the rest of the application code
COPY . /app

# Expose port 5000 for Flask
EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Use flask run to enable debugging
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]

# # The CMD will:
# # 1. Check if MODEL_NAME is phi3v.py and, if so, run the model script with the given epoch.
# # 2. Start the Flask application.
# CMD bash -c '\
#     if [ "$MODEL_NAME" = "phi3v.py" ]; then \
#         echo "Running phi3v.py with epoch ${EPOCH:-5}..."; \
#         python src/phi3v.py --epoch "${EPOCH:-5}"; \
#     else \
#         echo "MODEL_NAME is not set to phi3v.py; skipping model run."; \
#     fi; \
#     echo "Starting Flask app..."; \
#     exec python app.py'
