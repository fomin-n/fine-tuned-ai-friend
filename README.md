# fine-tuned-ai-friend

A Google Colab-ready notebook for fine-tuning a small language model with LoRA supervised fine-tuning on a Telegram chat export. The task is simple: given recent chat context, train the model to generate the selected friend's next message.

The default model is now `HuggingFaceTB/SmolLM2-360M-Instruct` because it is much less likely to run out of memory on a Colab T4 than Qwen's larger-vocabulary models. If you have a larger GPU, the notebook configuration cell shows how to switch back to `Qwen/Qwen3-0.6B`.

This project is intended for personal and private experimentation. Telegram exports, generated training examples, trained adapters, and model outputs may contain private information. Do not publish them unless every participant has consented and the content has been reviewed.

## What It Produces

- A trained LoRA adapter in the configured `OUTPUT_DIR`, defaulting to `smollm2-360m-ai-friend-lora/`
- Safe aggregate dataset statistics
- Manual generation examples
- A small Gradio chat demo running inside the Colab session

## How It Works

The notebook parses a Telegram `result.json`, asks you to choose which speaker should be modeled, builds supervised examples from recent chat history, fine-tunes the configured model with PEFT LoRA, and reloads the adapter for inference.

The notebook never includes real Telegram messages in the repository. It also avoids printing raw training examples by default.

## Usage

1. Export a Telegram chat as JSON from Telegram Desktop.
2. Put `result.json` in the notebook working directory. In Colab, upload it to the root of the session files.
3. Open `fine_tune_ai_friend_qwen3_0_6b_lora.ipynb` in Google Colab.
4. Select a GPU runtime.
5. Run the cells from top to bottom.
6. Choose the target friend speaker when prompted.
7. Train the LoRA adapter.
8. Use the generation helper or Gradio chat demo at the end.

## Telegram Export

In Telegram Desktop:

1. Open the chat.
2. Choose export chat history.
3. Select JSON format.
4. Save the export.
5. Copy or upload only `result.json` into the notebook runtime.

Do not commit `result.json`. The `.gitignore` is configured to keep Telegram exports, generated datasets, checkpoints, adapters, logs, caches, and local environment files out of Git.

## Files

- `fine_tune_ai_friend_qwen3_0_6b_lora.ipynb`: main Colab notebook
- `requirements.txt`: notebook dependencies
- `.gitignore`: privacy and generated artifact ignore rules

## Notes

- Training is intended for Google Colab GPU. Local execution may work for parsing and dataset preparation, but full fine-tuning is not expected to be practical on most CPUs.
- The notebook uses memory-conscious Colab defaults: `SmolLM2-360M-Instruct`, 4-bit loading, LoRA, sequence length 256, answer-only loss, train batch size 1, and no validation by default. If Colab still runs out of memory, restart the runtime, lower `MAX_SEQ_LENGTH`, lower `MAX_CONTEXT_MESSAGES`, lower `MAX_ANSWER_TOKENS`, or switch to `HuggingFaceTB/SmolLM2-135M-Instruct`.
- The quality of the generated replies depends on the amount and consistency of the Telegram export.
- Generated replies are synthetic and may be inaccurate, inappropriate, or unlike the selected person. Review outputs carefully.
