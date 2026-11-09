FROM python:3.11-slim

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libnuma1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "disgrpo.train"]
