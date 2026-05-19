# fine-tuned-ai-friend

A Kaggle-ready notebook for fine-tuning a small language model with LoRA supervised fine-tuning on a Telegram chat export. The task is simple: given recent chat context, train the model to generate the selected friend's next message.

The notebook is model-agnostic. It includes presets for `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-0.5B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`, and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

This project is intended for personal and private experimentation. Telegram exports, generated training examples, trained adapters, notebook outputs, and generated replies may contain private information. Do not publish them unless every participant has consented and the content has been reviewed.

## What It Produces

- A trained LoRA adapter in `/kaggle/working/friend_lora_output`
- Tokenizer files needed for inference
- Safe aggregate dataset statistics
- Synthetic-context generation examples saved to `/kaggle/working/generation_examples.json`
- Plain Python chat helpers and an optional Gradio chat UI

## How It Works

The notebook parses a Telegram `result.json`, asks you to choose which speaker should be modeled, builds supervised examples from recent chat history, fine-tunes the selected model with PEFT LoRA, saves the adapter, reloads it, and provides generation/chat helpers.

The notebook avoids printing raw Telegram messages by default. It prints aggregate statistics and a redacted example structure only.

## Usage On Kaggle

1. Export a Telegram chat as JSON from Telegram Desktop.
2. Create a private Kaggle Dataset containing `result.json`.
3. Attach that Dataset to the Kaggle notebook.
4. Open `friend_message_lora_sft_kaggle.ipynb`.
5. Enable a GPU accelerator, preferably P100.
6. Run the file-listing cell and set `DATA_PATH` if auto-detection does not find `result.json`.
7. Choose `MODEL_PRESET` if you want a different base model.
8. Run the notebook from top to bottom.
9. Choose the target friend speaker when prompted, or set `TARGET_SPEAKER_INDEX` in the configuration cell.
10. Download the adapter and generated examples from Kaggle outputs.

## Telegram Export

In Telegram Desktop:

1. Open the chat.
2. Choose export chat history.
3. Select JSON format.
4. Save the export.
5. Upload only `result.json` to a private Kaggle Dataset.

Do not commit `result.json`. The `.gitignore` is configured to keep Telegram exports, generated datasets, checkpoints, adapters, logs, caches, and local environment files out of Git.

## Files

- `friend_message_lora_sft_kaggle.ipynb`: main Kaggle notebook
- `requirements.txt`: notebook dependencies
- `.gitignore`: privacy and generated artifact ignore rules

## Notes

- The notebook is designed for Kaggle GPU sessions and uses Kaggle input and working directories.
- P100-compatible defaults use fp16, no bf16, no FlashAttention assumptions, 4-bit loading, LoRA, conservative batch size, gradient accumulation, and answer-only loss.
- If the P100 runs out of memory, restart the session and switch to `smollm2_360m_instruct`, reduce `MAX_CONTEXT_MESSAGES`, reduce `MAX_ANSWER_TOKENS`, or choose a shorter preset sequence length.
- The quality of generated replies depends on the size and consistency of the Telegram export.
