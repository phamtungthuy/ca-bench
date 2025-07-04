FROM python:3.10-slim

WORKDIR /app

# Install dependencies for OpenCV, OpenGL and PyAv
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    git-lfs \
    ffmpeg \
    nginx \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./server /app/server
COPY ./data /app/data

COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh && sed -i 's/\r$//' ./entrypoint.sh

ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]
