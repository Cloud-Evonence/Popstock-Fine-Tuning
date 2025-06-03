from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
from google.cloud import storage
# Google Drive and GCS configuration
drive_file_id = "your-drive-file-id" # Replace with your Google Drive file ID (e.g., from a shared link)
gcs_bucket_name = "llama2fine_tune"
gcs_blob_name = "popstock.jsonl"
credentials_path = "/path/to/service-account-key.json" # Replace with your service account key path or use default auth
# Authenticate with Google Drive
credentials = service_account.Credentials.from_service_account_file(
credentials_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build("drive", "v3", credentials=credentials)
# Download file from Google Drive
request = drive_service.files().get_media(fileId=drive_file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
status, done = downloader.next_chunk()
print(f"Download {int(status.progress() * 100)}%.")
# Read and process the downloaded file (assume text or CSV with Q&A pairs)
data = []
fh.seek(0)
content = fh.read().decode("utf-8")
# Assume the file is a text file with lines like "Question: ... Answer: ..."
if content.startswith("Question:") or content.startswith("Answer:"):
# Parse text format (e.g., "Question: What is X? Answer: Y.")
lines = content.strip().split("\n")
for line in lines:
if "Question:" in line and "Answer:" in line:
question = line.split("Question:")[1].split("Answer:")[0].strip()
answer = line.split("Answer:")[1].strip()
data.append({"input_text": question, "output_text": answer})
elif "," in content: # Assume CSV format
import csv
from io import StringIO
csv_reader = csv.DictReader(StringIO(content))
for row in csv_reader:
data.append({"input_text": row.get("input_text", ""), "output_text": row.get("output_text", "")})
else:
raise ValueError("Unsupported file format. Use text with 'Question: ")
# Convert to JSONL format
jsonl_content = "\n".join([json.dumps(item) for item in data])
# Upload to GCS
storage_client = storage.Client()
bucket = storage_client.bucket(gcs_bucket_name)
blob = bucket.blob(gcs_blob_name)
blob.upload_from_string(jsonl_content, content_type="application/jsonl")
print(f"Data converted to JSONL and uploaded to gs://{gcs_bucket_name}/{gcs_blob_name}")
# Verify upload
blob = bucket.get_blob(gcs_blob_name)
print(f"File size in GCS: {blob.size} bytes")