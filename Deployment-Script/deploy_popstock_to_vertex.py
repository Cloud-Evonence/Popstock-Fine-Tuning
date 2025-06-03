from google.cloud import storage
from google.cloud import aiplatform
import os
# Model configuration
new_model = "Llama-2-7b-popstock-finetune"
project_id = "your-project-id" # Replace with your GCP project ID
bucket_name = "llama2fine_tune" # Use your existing bucket
# Upload to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
for root, _, files in os.walk(new_model):
for file in files:
local_path = os.path.join(root, file)
blob_path = os.path.join("finetuned_llama2_popstock", os.path.relpath(local_path, new_model))
blob = bucket.blob(blob_path)
blob.upload_from_filename(local_path)
print(f"Model uploaded to gs://{bucket_name}/finetuned_llama2_popstock")
# Deploy to Vertex AI
aiplatform.init(project=project_id, location="us-central1")
model_display_name = "finetuned-llama2-popstock-7b"
model_path = f"gs://{bucket_name}/finetuned_llama2_popstock"
container_uri = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240222_0916_RC00"
model = aiplatform.Model.upload(
display_name=model_display_name,
artifact_uri=model_path,
serving_container_image_uri=container_uri,
serving_container_args=["--model", "/model", "--port", "8080"],
)
endpoint = model.deploy(
machine_type="n1-standard-8",
accelerator_type="NVIDIA_TESLA_T4",
accelerator_count=1,
min_replica_count=1,
max_replica_count=1,
)
print(f"Endpoint created: {endpoint.resource_name}")