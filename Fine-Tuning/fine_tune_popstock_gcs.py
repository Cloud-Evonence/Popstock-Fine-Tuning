import torch 
import json 
from google.cloud import storage 
from transformers import ( 
AutoModelForCausalLM, 
AutoTokenizer, 
BitsAndBytesConfig, 
TrainingArguments, 
DataCollatorForLanguageModeling, 
) 
from peft import LoraConfig, get_peft_model 
from trl import SFTTrainer 
from datasets import Dataset 
import os 

# Set PyTorch memory management flag 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

# Model and dataset configuration 
model_name = "meta-llama/Llama-2-7b-hf" 
gcs_bucket_path = "gs://llama2fine_tune/popstock.jsonl" # Your GCS path 
new_model = "Llama-2-7b-popstock-finetune" 

# QLoRA parameters 
lora_r = 32 
lora_alpha = 32 # Increased to boost adaptation 
lora_dropout = 0.05 # Reduced dropout for better learning 

# BitsAndBytes parameters 
use_4bit = True 
bnb_4bit_compute_dtype = "float16" 
bnb_4bit_quant_type = "nf4" 
use_nested_quant = False 

# TrainingArguments parameters 
output_dir = "./results_popstock_gcs" 
num_train_epochs = 5 # Increased for better adaptation 
fp16 = True # T4 supports fp16 
bf16 = False # T4 doesn’t support bf16 
per_device_train_batch_size = 1 
per_device_eval_batch_size = 1 
gradient_accumulation_steps = 4 
gradient_checkpointing = True 
max_grad_norm = 0.3 
learning_rate = 1.5e-4 # Slightly reduced for stability 
weight_decay = 0.001 
optim = "paged_adamw_32bit" 
lr_scheduler_type = "cosine" 
max_steps = -1 
warmup_ratio = 0.03 
group_by_length = True 
save_steps = 50 
logging_steps = 10 

# Define max sequence length for preprocessing 
max_seq_length = 256 # Optimized for T4 memory 
device_map = {"": 0} 

# Load dataset from GCS 
def load_dataset_from_gcs(bucket_path): 
storage_client = storage.Client() 
bucket_name = bucket_path.split("/")[2] 
blob_name = "/".join(bucket_path.split("/")[3:]) 
bucket = storage_client.bucket(bucket_name) 
blob = bucket.blob(blob_name) 

# Download and read the JSONL file 
data = [] 
content = blob.download_as_text() 
for line in content.splitlines(): 
if line.strip(): 
try: 
entry = json.loads(line) 
data.append({"input_text": entry.get("input_text", ""), "output_text": entry.get("output_text", "")}) 
except json.JSONDecodeError as e: 
print(f"Error decoding line: {line}, {e}") 
continue 
return Dataset.from_list(data) 
https://colab.research.google.com/drive/1VrumKYbjtmAKZ6tY1ERKU7H-wFvKK7aF#printMode=true 1/3

# Load dataset 
try: 
dataset = load_dataset_from_gcs(gcs_bucket_path) 
print(f"Dataset loaded successfully from {gcs_bucket_path}") 
except Exception as e: 
print(f"Error loading dataset from GCS: {e}") 
raise 

# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right" 

# Format dataset for Q&A pairs (instruction format for LLaMA-2) 
def preprocess_function(examples): 
instructions = [f"[INST] {q} [/INST]" for q in examples["input_text"]] 
responses = examples["output_text"] 
full_texts = [f"{inst} {resp}" for inst, resp in zip(instructions, responses)] 
model_inputs = tokenizer(full_texts, truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="pt") model_inputs["labels"] = model_inputs["input_ids"].clone() # Set labels for causal LM 
return model_inputs 
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "output_text"]) 

# Set up quantization config 
compute_dtype = getattr(torch, bnb_4bit_compute_dtype) 
bnb_config = BitsAndBytesConfig( 
load_in_4bit=use_4bit, 
bnb_4bit_quant_type=bnb_4bit_quant_type, 
bnb_4bit_compute_dtype=compute_dtype, 
bnb_4bit_use_double_quant=use_nested_quant, 
) 

# Load base model 
model = AutoModelForCausalLM.from_pretrained( 
model_name, 
quantization_config=bnb_config, 
device_map=device_map, 
trust_remote_code=True, 
) 
model.config.use_cache = False 
model.config.pretraining_tp = 1 

# Load LoRA configuration 
peft_config = LoraConfig( 
lora_alpha=lora_alpha, 
lora_dropout=lora_dropout, 
r=lora_r, 
bias="none", 
task_type="CAUSAL_LM", 
) 

# Set training parameters 
training_arguments = TrainingArguments( 
output_dir=output_dir, 
num_train_epochs=num_train_epochs, 
per_device_train_batch_size=per_device_train_batch_size, 
gradient_accumulation_steps=gradient_accumulation_steps, 
optim=optim, 
save_steps=save_steps, 
logging_steps=logging_steps, 
learning_rate=learning_rate, 
weight_decay=weight_decay, 
fp16=fp16, 
bf16=bf16, 
max_grad_norm=max_grad_norm, 
max_steps=max_steps, 
warmup_ratio=warmup_ratio, 
group_by_length=group_by_length, 
lr_scheduler_type=lr_scheduler_type, 
report_to="tensorboard", 
) 
https://colab.research.google.com/drive/1VrumKYbjtmAKZ6tY1ERKU7H-wFvKK7aF#printMode=true 2/3

# Set up data collator 
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) 

# Initialize SFTTrainer (trl 0.15.1 compatible) 
trainer = SFTTrainer( 
model=model, 
train_dataset=tokenized_dataset, 
peft_config=peft_config, 
tokenizer=tokenizer, 
args=training_arguments, 
data_collator=data_collator, 
) 

# Train model 
trainer.train() 

# Save the fine-tuned model 
trainer.model.save_pretrained(new_model) 
tokenizer.save_pretrained(new_model) 
print(f"Fine-tuning completed. Model saved to {new_model}") 
https://colab.research.google.com/drive/1VrumKYbjtmAKZ6tY1ERKU7H-wFvKK7aF#printMode=true 3/3
