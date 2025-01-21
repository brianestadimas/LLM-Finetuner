# Dockerfile

FROM python:3.12-slim

# Create working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

COPY . /app

# Expose port 5000 for Flask
EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Use flask run to enable debugging
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]
