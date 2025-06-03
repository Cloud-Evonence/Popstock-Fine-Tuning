# Popstock-Fine-Tuning
Step 1: GDrive-to-GCS
 This section converts text data (e.g., Q&A pairs in a text file, CSV, or Google Docs) from Google Drive into JSONL format and uploads it to gs://llama2fine_tune/popstock.jsonl.
 Find end to end script for the same: prepare_data_gdrive_to_gcs.py
 
Step 2: Fine-Tuning
 This script fine-tunes LLaMA-2-7B on the popstock.jsonl dataset from GCS using QLoRA.
 Find end to end script for the same: fine_tune_popstock_gcs

Step 3: Testing-Script
 This script loads the fine-tuned model and tests it with 8 prompts from the popstock.jsonl dataset, ensuring non-repetitive, relevant responses.
 Find end to end script for the same: test_popstock_gcs.py

Step 4: Deployment-Script
 This script uploads the fine-tuned model to GCS and deploys it to Vertex AI for inference.
 Find end to end script for the same: deploy_popstock_to_vertex.py
