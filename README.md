# RX + Unsloth Vision Fine-Tuning

Production-ready pipeline for fine-tuning and evaluating Llama-3.2 Vision models on the RX petrographic thin-section dataset.

## Project Status

- Data conversion pipeline: complete and stable.
- Unsloth training pipeline (script + notebook): complete and runnable.
- Post-training evaluation notebook with model picker + interactive inference: complete.
- Colab + local workflows: both supported.
- Current scope: non-taxonomy rock-type classification.

## Final Metrics (Latest Recorded Runs)

These are the latest metrics captured from executed notebook outputs in this repo.

### Full Test Evaluation (1120 examples)
- Model: `rx_lora_20260226_004614`
- Source: `RX_BASE_Model_Evaluation.ipynb`
- Accuracy: `0.9116`
- F1 macro: `0.9112`
- F1 weighted: `0.9112`
- Precision macro: `0.9122`
- Recall macro: `0.9116`
- Empty/unknown predictions: `0`

### Training Notebook Eval Snapshot (100 examples)
- Source: `RX_BASE_Unsloth_Vision_Finetune.ipynb`
- Accuracy: `0.93`
- F1 macro: `0.9026`
- F1 weighted: `0.9264`
- Precision macro: `0.9420`
- Recall macro: `0.8915`
- Empty/unknown predictions: `0`

## Repository Layout

- `prepare_rx_for_unsloth.py`: converts RX JSON (`image` absolute paths + `conversations`) to clean path-stable JSON.
- `train_unsloth_vision.py`: standalone training entrypoint.
- `RX_BASE_Unsloth_Vision_Finetune.ipynb`: notebook training flow.
- `RX_BASE_Model_Evaluation.ipynb`: standalone post-training evaluation + interactive inference.
- `requirements.txt`: minimal local dependency list.
- `data/train_clean.json`, `data/test_clean.json`: converted datasets.

## Data Format

### Input (RX)
```json
{
  "image": "/absolute/path/to/104_01311.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nClassify this rock type."},
    {"from": "gpt", "value": "Diorite"}
  ]
}
```

### Converted (`prepare_rx_for_unsloth.py`)
```json
{
  "image": "104_01311.jpg",
  "question": "Classify this rock type.",
  "answer": "Diorite"
}
```

### Training Dataset Shape (Unsloth)
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/RX_BASE/Imagenes/104_01311.jpg"},
        {"type": "text", "text": "Classify this rock type."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "Diorite"}]
    }
  ]
}
```

## Quick Start

### 1) Prepare Clean JSON
```bash
python RX_unsloth/prepare_rx_for_unsloth.py \
  --train-json RX/train_data.json \
  --test-json RX/test_data.json \
  --out-dir RX_unsloth/data
```

### 2) Train (Script)
```bash
python RX_unsloth/train_unsloth_vision.py \
  --train-json RX_unsloth/data/train_clean.json \
  --images-dir /path/to/RX_BASE/Imagenes \
  --model unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit \
  --output-dir RX_unsloth/outputs \
  --max-steps 350
```

### 3) Evaluate (Notebook)

Open `RX_unsloth/RX_BASE_Model_Evaluation.ipynb` and run all cells.

## Recommended Settings

- Model: `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
- `SUBSET=0` (full data)
- `EPOCHS=2` to `3`
- `LEARNING_RATE=1e-4`
- `EVAL_TEMPERATURE=0.05` for stable classification outputs

## Critical Constraints

Unsloth vision SFT needs these settings:

- `remove_unused_columns=False`
- `dataset_text_field=""`
- `dataset_kwargs={"skip_prepare_dataset": True}`

These are already set in both notebook and script training flows.

## Troubleshooting

### Meta Tensor Error
`RuntimeError: Cannot copy out of meta tensor; no data!`

Cause: loading very large non-4bit model variants.

Fix: use `*-bnb-4bit` checkpoints.

### Image Not Found During Eval/Train

Cause: `IMAGES_DIR` is wrong for the current machine/runtime.

Fix: point to the extracted image folder containing files like `104_01311.jpg`.

### OOM

1. Lower `BATCH_SIZE` (e.g. 2 -> 1)
2. Lower `GRAD_ACCUM`
3. Use 11B model instead of larger variants
