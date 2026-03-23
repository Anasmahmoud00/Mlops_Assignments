FROM python:3.10-slim

WORKDIR /app

# Accept RUN_ID as a build argument
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Simulate downloading the model from MLflow
RUN echo "Downloading model for Run ID: ${RUN_ID} from MLflow..." > model_status.txt

CMD ["python", "train.py"]
