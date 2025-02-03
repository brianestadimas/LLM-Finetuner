FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the C compiler explicitly
ENV CC=gcc

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Copy the entire application
COPY . /app

# Run the patch script during the build process
RUN if [ -f "patch/patch_builder.sh" ]; then \
    echo "Running patch/patch_builder.sh"; \
    bash patch/patch_builder.sh; \
    else \
    echo "patch/patch_builder.sh not found, skipping..."; \
    fi

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Use flask run to enable debugging
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]