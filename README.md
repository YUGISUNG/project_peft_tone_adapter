# PEFT Tone Adapter (LoRA + Flan-T5-Small)

This project demonstrates a parameter-efficient fine-tuning (PEFT) approach using [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) to train a lightweight tone-rewriting adapter on top of the frozen `flan-t5-small` model.

The adapter specializes in:
- Rewriting sentences into a professional tone
- Encouraging tone transformations
- Grammar corrections
- Casual summaries and stylistic rewrites

---

## 🔧 Built With

- 🤗 `transformers`
- 🧠 [`peft`](https://github.com/huggingface/peft)
- 🗂️ `datasets`
- 🏋️‍♂️ `flan-t5-small` as the base model
- 🐍 Python 3.10+ and `torch`

---

## 📁 Folder Structure

```
project_peft_tone_adapter/
├── data/                         # Training dataset
│   └── tone_dataset.json
├── results/                      # Trained adapter output
│   └── lora_adapter/
├── train/
│   ├── train_lora_flan.py        # PEFT training script
│   ├── inference_lora_flan.py    # Inference script
├── .gitignore
└── README.md
```

---

## 📌 How to Use

### ✅ 1. Install Dependencies

In a clean Python or Conda environment:

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

> Note: bitsandbytes is optional on CPU-only setups.

---

### ✅ 2. Train the Adapter

Make sure your dataset is located at `data/tone_dataset.json`.

Then run:

```bash
python train/train_lora_flan.py
```

This will train a LoRA adapter on the frozen `flan-t5-small` and save it to `results/lora_adapter/`.

---

### ✅ 3. Run Inference

After training, test the adapter by running:

```bash
python train/inference_lora_flan.py
```

You can modify the `prompt =` line in the script to try different inputs.

---

## 🗣 Example Prompt

```text
Rewrite this in a professional tone:
Yo I can't help you with that rn.
```

**Model Response:**
```
I'm sorry, but I can't help you with that at the moment.
```

---

## 💡 Why LoRA?

LoRA lets you add new behaviors to large language models without updating or retraining the full model. It’s fast, memory-efficient, and highly modular.

In this project, we injected tone-specific behaviors into a frozen model using only ~0.1% of the parameters.

---

## 🧠 Future Improvements

- Add UI with Streamlit or Gradio
- Expand dataset with more tones (casual, sarcastic, empathetic, etc.)
- Swap `flan-t5-small` for larger models (e.g., `flan-t5-base` or `mistral-7b`)

---

## 🧑‍💻 Author

Built by JP, powered by curiosity and LoRA ✨  
