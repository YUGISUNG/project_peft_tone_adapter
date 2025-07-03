import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load dataset
dataset = load_dataset("json", data_files="../data/tone_dataset.json")["train"]

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

# Preprocess
def preprocess(example):
    prompt = f"Rewrite this in a {example['tone']} tone: {example['input']}"
    model_input = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)
    with tokenizer.as_target_tokenizer():
        label = tokenizer(example["output"], truncation=True, padding="max_length", max_length=128)
    model_input["labels"] = label["input_ids"]
    return model_input

tokenized_dataset = dataset.map(preprocess)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="../results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save adapter
model.save_pretrained("../results/lora_adapter")
