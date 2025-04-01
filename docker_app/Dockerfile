FROM python:3.12-slim

# Create working directory
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     build-essential \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    gnupg \
    ca-certificates \
    libnss3 \
    libgdk-pixbuf2.0-0 \
    libxss1 \
    libappindicator3-1 \
    libatk-bridge2.0-0 \
    libasound2 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && apt-get install -y google-chrome-stable

# Install ChromeDriver
RUN CHROME_DRIVER_VERSION=`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE` && \
    wget https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip -d /usr/local/bin/ && \
    rm chromedriver_linux64.zip

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