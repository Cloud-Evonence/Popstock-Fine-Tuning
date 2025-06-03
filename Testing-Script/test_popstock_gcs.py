import torch
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
pipeline,
)
import os
# Set PyTorch memory management flag
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Model configuration
new_model = "Llama-2-7b-popstock-finetune"
# BitsAndBytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
# Define device mapping
device_map = {"": 0}
# Set up quantization config
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
load_in_4bit=use_4bit,
bnb_4bit_quant_type=bnb_4bit_quant_type,
bnb_4bit_compute_dtype=compute_dtype,
bnb_4bit_use_double_quant=use_nested_quant,
)
# Load the fine-tuned model and tokenizer
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
new_model,
quantization_config=bnb_config,
device_map=device_map,
trust_remote_code=True,
)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(new_model, trust_remote_code=True)
# Create a text generation pipeline
generator = pipeline(
"text-generation",
model=fine_tuned_model,
tokenizer=fine_tuned_tokenizer,
)
# Test with specific prompts from the dataset (first 8 for variety)
test_prompts = [
"What is the main topic discussed in 'artist-earnings-tracking'?",
"What key insights are provided in 'artist-earnings-tracking'?",
"How does 'artist-earnings-tracking' contribute to understanding music markets?",
"What is the main topic discussed in 'derivative-pricing-models'?",
"What key insights are provided in 'derivative-pricing-models'?",
"How does 'derivative-pricing-models' contribute to understanding music markets?",
"What are the main topics discussed in the documents?",
"How do the documents contribute to understanding economic aspects of music?",
]
formatted_prompts = [f"[INST] {prompt} [/INST]" for prompt in test_prompts]
# Generate and compare responses (using expected outputs from your previous data)
expected_outputs = [
"The document ‘artist-earnings-tracking’ discusses understanding artist earnings through Travis Scott’s return to festival performanc
"Key insights include the Astroworld incident. This case study...",
"It contributes by explaining artist earnings: Streaming Revenue: Michael noticed that despite controversy, Scott’s streaming...",
"The document ‘derivative-pricing-models’ discusses Modeling Tomorrow’s Music Value Michael’s breakthrough in derivative...",
"Key insights include bank, he discovered that traditional Black-Scholes pricing models failed to capture the unique...",
"It contributes by explaining or the episodic nature of artistic creation and the non-normal distribution of success outcomes...",
"The main topics are artist earnings tracking and derivative pricing models in music.",
"The documents contribute by analyzing economic aspects like artist earnings and pricing models in music markets.",
]

https://colab.research.google.com/drive/1VrumKYbjtmAKZ6tY1ERKU7H-wFvKK7aF#scrollTo=WM2JQYCpuCU9&printMode=true 1/2

# Generate and compare responses
print("\n=== Testing the Fine-Tuned Model ===")
for prompt, expected_output in zip(formatted_prompts, expected_outputs):
print(f"\nPrompt: {prompt}")
output = generator(
prompt,
max_new_tokens=200,
do_sample=True,
top_k=30,
top_p=0.9,
temperature=0.5,
)
response = output[0]["generated_text"]
# Remove the prompt from the response if present
if response.startswith(prompt):
response = response[len(prompt):].strip()
# Truncate response to first sentence for clarity
response = response.split(".")[0].strip() + "." if "." in response else response
print(f"Generated Response: {response}")
print(f"Expected Output: {expected_output}")