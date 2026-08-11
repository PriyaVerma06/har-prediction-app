# Production Dockerfile for Hugging Face Spaces (Docker SDK)
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user

# Set up working directory in user's home
WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code with proper ownership
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Expose Hugging Face Space default port
EXPOSE 7860

# Run Uvicorn server binding to 0.0.0.0 and dynamically reading $PORT (default 7860)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
