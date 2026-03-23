import mlflow
client = mlflow.tracking.MlflowClient()
# Replace '...' with the actual experiment name if known, but based on the image:
experiment_name = "Assignment3_Anas"
exp = client.get_experiment_by_name(experiment_name)
if exp is None:
    print(f"Experiment '{experiment_name}' not found.")
    import sys
    sys.exit(1)

runs = client.search_runs(
    exp.experiment_id,
    order_by=['metrics.d_accuracy DESC'], # In the image it was metrics.accuracy, but here we have d_accuracy
    max_results=1
)

if not runs:
    print("No runs found.")
    import sys
    sys.exit(1)

with open('best_model_uri.txt', 'w') as f:
    f.write(runs[0].info.artifact_uri + '/model')

print(f"Best model URI saved to best_model_uri.txt: {runs[0].info.artifact_uri}/model")
