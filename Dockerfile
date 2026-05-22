FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .
COPY src/ src/
COPY image.jpg .

CMD ["python", "run.py", "--input_dir", "/app/input_images", "--output_dir", "/app/output_images"]
