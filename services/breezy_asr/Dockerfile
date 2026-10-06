FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY warmup.wav .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
