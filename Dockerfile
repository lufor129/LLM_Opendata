FROM python:3.9-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip3 install -r requirements.txt

# 複製應用代碼
COPY . .

# 暴露端口
EXPOSE 8080

# 設置環境變量
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# 創建啟動腳本
RUN echo '#!/bin/bash\n\
echo "Starting data processing..."\n\
python csv_to_rag.py &\n\
sleep 5\n\
echo "Starting Streamlit server..."\n\
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.maxUploadSize 200' > /app/start.sh \
    && chmod +x /app/start.sh

# 使用啟動腳本
CMD ["/app/start.sh"]