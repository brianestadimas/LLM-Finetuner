# Use the official Python 3.11 image from Docker Hub
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port your app will run on (optional, for Flask, default is 5000)
EXPOSE 5000

# Run your app (adjust the command based on your project)
CMD ["python", "app.py"]
