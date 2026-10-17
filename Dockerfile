FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV MODELS_DIR=/app/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
# Verify the upload parser is present in the locked environment.
RUN python -c "import python_multipart"

# .dockerignore allows runtime code only; no videos, secrets, results or weights.
COPY . .
RUN useradd --create-home service && mkdir -p runs/jobs models && chown -R service:service runs models
USER service

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)"
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
