FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy trained model (produced by src/train.py)
COPY data/processed/risk_model.pkl data/processed/risk_model.pkl

ENV MODEL_PATH=data/processed/risk_model.pkl
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
