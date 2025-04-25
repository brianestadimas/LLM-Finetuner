FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     build-essential \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    poppler-utils \
    wget \
    curl \
    unzip \
    ca-certificates \
    gnupg \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome the proper way using keyring (replaces old apt-key/deb method)
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /etc/apt/keyrings/google.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Confirm Chrome location is valid (usually /usr/bin/google-chrome)
RUN which google-chrome

# Set the C compiler explicitly
ENV CC=gcc
RUN mkdir -p /app/lancedb

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN pip install olive-ai[ort-genai,auto-opt]
RUN pip install --upgrade bitsandbytes

# Copy the entire application
COPY . /app

# Run the patch script during the build process
RUN if [ -f "patch/patch_builder.sh" ]; then \
    echo "Running patch/patch_builder.sh"; \
    bash patch/patch_builder.sh || echo "patch_builder.sh encountered an error, skipping..."; \
    else \
    echo "patch/patch_builder.sh not found, skipping..."; \
    fi


# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV LANCEDB_URI ""
ENV LANCEDB_API_KEY ""
ENV LANCE_API_KEY ""
ENV LANCE_URI ""
RUN unset LANCE_API_KEY

# Use flask run to enable debugging
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "4", "--worker-class", "sync"]