# 1. Load required libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# 2. Load the trained LoRA config — this tells us where the adapter is
peft_model_path = "../results/lora_adapter"  # Folder created during training
config = PeftConfig.from_pretrained(peft_model_path)

# 3. Load the base model and tokenizer (flan-t5-small)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)

# 4. Load the LoRA adapter and attach it to the base model
model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()  # Set to inference mode

# 5. Define your custom prompt here
prompt = "Make this sound encouraging:\nYou didn’t pass the test."
# 6. Tokenize the input and run the model
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

# 7. Decode and print the response
print("\n🔁 Model response:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
