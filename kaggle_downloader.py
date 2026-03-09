import kagglehub

# Download latest version
path = kagglehub.dataset_download("pervolarakis/bigearth-federated-few-shot")

print("Path to dataset files:", path)
