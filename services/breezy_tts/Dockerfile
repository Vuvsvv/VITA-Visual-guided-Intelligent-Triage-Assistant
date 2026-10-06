FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# 先 clone BreezyVoice repo（含子模組）
RUN git clone https://github.com/mtkresearch/BreezyVoice.git /app/BreezyVoice \
    && cd /app/BreezyVoice \
    && pip install --no-cache-dir -r requirements.txt

# 安裝 API 服務依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
ENV BREEZY_REPO_PATH=/app/BreezyVoice
ENV BREEZY_MODEL_PATH=MediaTek-Research/BreezyVoice

EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
