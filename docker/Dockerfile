# Dockerfile for MLP Churn Classifier API
# Build context must be the PROJECT ROOT (one level above docker/).
# Run: docker build -f docker/Dockerfile -t churn-api .
#   or: docker compose up  (from docker/ with the fixed docker-compose.yml)

# Stage 1: Builder — install all Python deps
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime — lean image with only what's needed
FROM python:3.10-slim

WORKDIR /app

# Bring in the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source maintaining the directory structure that serving/app.py depends on:
#   serving/app.py resolves ROOT_DIR = Path(__file__).resolve().parents[1]  -> /app
#   then adds /app/src to sys.path for model imports
COPY src/ ./src/
COPY serving/ ./serving/
COPY artifacts/ ./artifacts/

EXPOSE 5000

# Built-in Python urllib avoids a `requests` dependency in the slim image
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Launch the FastAPI app via uvicorn (consistent with local `uvicorn serving.app:app`)
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "5000"]
