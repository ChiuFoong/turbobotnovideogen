# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables for a smoother build and run experience
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - Use --no-install-recommends to keep the image slim
# - Chain commands to reduce layers and clean up apt cache in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker's layer caching
# This layer only gets rebuilt if requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
# FIX: Upgrade pip and use the legacy resolver to bypass dependency conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a non-root user for security and set ownership
RUN useradd -m -s /bin/bash streamlit && chown -R streamlit:streamlit /app

# Switch to the non-root user
USER streamlit

# Expose the port the Streamlit app will run on
EXPOSE 8501

# Define the command to run your application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
