[canbingol/tr-gemma-3-270m-it](https://huggingface.co/canbingol/tr-gemma-3-270m-it)

# Fine-tuned Gemma-3 270M for Turkish

This repository provides a fine-tuned version of `google/gemma-3-270m-it` on the Turkish dataset `ucekmez/OpenOrca-tr`. Fine-tuning was done using TRL's `SFTTrainer`.

## Training Configuration

- Base model: google/gemma-3-270m-it
- Max sequence length: 512
- Batch size: 14
- Learning rate: 5e-5
- Optimizer: AdamW (torch fused)
- Precision: auto (bf16 used)
- Number of epochs: 5
- Logging steps: 1
- Save strategy: every 5000 steps
- Eval strategy: every 1000 steps
- Dataset split: 99.8% train / 0.2% eval (2k sample)

## Evaluation Results

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 1000 | 2.7863     | 2.4349    |
| 2000 | 2.6299     | 2.3219    |
| 3000 | 2.1846     | 2.2640    |
| 4000 | 2.4088     | 2.2216    |
| 5000 | 2.2412     | 2.1895    |

## Notes

- Fine-tuning was performed in Colab.
- Tokenizer and model were pushed using `huggingface_hub.upload_folder`.
- Model supports chat prompt via `apply_chat_template`.
