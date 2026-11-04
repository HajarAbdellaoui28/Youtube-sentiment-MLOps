
FROM python:3.12-slim

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ ./src/
COPY app_api.py .
COPY models/ ./models/

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT:-7860}"]
