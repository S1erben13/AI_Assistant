FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV CONFIGS_DIR=/app/configs
ENV DOCKER_MODE=true
ENV CONFIGS_DIR=/app/configs
ENV DATA_DIR=/app/data

#CMD [ "pytest", "tests/" ]
#CMD ["python", "scripts/load_vector_db.py"]