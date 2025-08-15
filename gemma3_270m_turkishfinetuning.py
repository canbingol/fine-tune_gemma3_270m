# Commented out IPython magic to ensure Python compatibility.
# %pip install -q torch tensorboard
# %pip install -q transformers datasets accelerate evaluate trl protobuf sentencepiece
# %pip install -q flash-attn

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login

# Login into Hugging Face Hub
login("api_key")

base_model = "google/gemma-3-270m-it"
checkpoint_dir = "/content/drive/MyDrive/gemma0.27-tr"
learning_rate = 5e-5

from datasets import load_dataset

def create_conversation(sample):
  return {
      "messages": [
          {"role": "user", "content": sample["question"]},
          {"role": "assistant", "content": sample["response"]}
      ]
  }


dataset = load_dataset("ucekmez/OpenOrca-tr", split="train")

dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

dataset = dataset.train_test_split(test_size=0.02, shuffle=False)

print(dataset["train"][0]["messages"])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")

from transformers import pipeline

from random import randint
import re

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

rand_idx = randint(0, len(dataset["test"])-1)
test_sample = dataset["test"][rand_idx]

prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

print(f"Question:\n{test_sample['messages'][0]['content']}\n")
print(f"Original Answer:\n{test_sample['messages'][1]['content']}\n")
print(f"Generated Answer (base model):\n{outputs[0]['generated_text'][len(prompt):].strip()}")

outputs = pipe([{"role": "user", "content": "merhaba"}], max_new_tokens=256, disable_compile=True)
print(outputs[0]['generated_text'][1]['content'])

from trl import SFTConfig

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=checkpoint_dir,
    max_length=512,
    packing=False,
    num_train_epochs=5,
    per_device_train_batch_size=14,
    gradient_checkpointing=False,
    optim="adamw_torch_fused",
    logging_steps=1,
    save_strategy="steps",
    save_steps=5000,
    eval_strategy="steps",
    eval_steps=1000,
    learning_rate=learning_rate,
    fp16=True if torch_dtype == torch.float16 else False,
    bf16=True if torch_dtype == torch.bfloat16 else False,
    lr_scheduler_type="constant",
    push_to_hub=True,
    report_to="tensorboard",
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True,
    }
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model()

from huggingface_hub import create_repo, upload_folder

repo_id = "canbingol/tr-gemma-3-270m-it"
create_repo(repo_id, exist_ok=True)

folder_to_push = checkpoint_dir

upload_folder(
    repo_id=repo_id,
    folder_path=folder_to_push,
    repo_type="model",
    commit_message="Initial upload"
)

