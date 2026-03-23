import mlflow
import sys
import os

# Read Run ID from model_info.txt
if not os.path.exists("model_info.txt"):
    print("model_info.txt not found.")
    sys.exit(1)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

try:
    # Fetch the run data
    run = client.get_run(run_id)
    metrics = run.data.metrics

    # Get accuracy (using d_accuracy as used in train.py)
    accuracy = metrics.get("d_accuracy", 0.0)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")

    threshold = 0.85
    if accuracy < threshold:
        print(f"FAILED: Accuracy {accuracy} is below threshold {threshold}")
        sys.exit(1)

    print(f"SUCCESS: Accuracy {accuracy} meets threshold {threshold}")
except Exception as e:
    print(f"Error checking threshold: {e}")
    sys.exit(1)
