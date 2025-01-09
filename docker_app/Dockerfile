# Dockerfile

FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 5000 for Flask
EXPOSE 5000

CMD ["python", "app.py"]

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
