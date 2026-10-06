FROM python:3.10-slim
WORKDIR /app

# ffmpeg 用於 m4a 轉檔，libsndfile1 給 soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY prompts ./prompts
COPY data ./data

# GPU 的部分在宿主機的 Chatterbox 引擎（engine/cb_server.py），本服務只是 HTTP client。
# engine/ 刻意不 COPY 進映像：它需要 GPU 與 chatterbox-finetuning 目錄，必須跑在宿主機。
ENV CB_API=http://host.docker.internal:9881

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
