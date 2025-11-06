FROM python:3.10-slim

# System deps for librosa/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    ffmpeg \ 
    libsndfile1 \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Default command shows help
CMD ["python", "cli.py", "--help"]
