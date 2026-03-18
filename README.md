[README.md](https://github.com/user-attachments/files/26093284/README.md)
# Meteorite Detection Network

This repository accompanies the manuscript *Parameter-Efficient Fine-Tuning of a Vision-Language Model for Rock Type Classification from Thin-Section Images* and currently contains the main training notebook, `RX-11B-Full-3E.ipynb`.

Despite the repository name, the work in this repository is a rock-type thin-section image classification project. The notebook fine-tunes a vision-language model to read a petrographic image and return only the predicted rock class.

## Manuscript summary

The paper reports a parameter-efficient fine-tuning workflow built on `unsloth/Llama-3.2-11B-Vision-Instruct` for 14-class rock classification from thin-section images.

Reported setup from the manuscript:

- dataset size: 5,600 images
- classes: rhyolite, andesite, basalt, granite, diorite, gabbro, ultramafic rock, phyllite, schist, gneiss, marble, sandstone, limestone, and shale
- lighting views: plane-polarized light and cross-polarized light
- train/test split: 4,480 train and 1,120 test images
- quantization: 4-bit
- PEFT method: LoRA with rank `32` and `alpha=16`
- training schedule: `1,680` steps over `3` epochs
- effective batch size: `8` (`batch_size=2`, `gradient_accumulation=4`)
- trainable share: about `1.24%` of total parameters

Reported test-set results from the manuscript:

- accuracy: `91.2%`
- macro-F1: `0.9112`
- weighted-F1: `0.9112`
- macro precision: `0.9122`
- macro recall: `0.9116`
- unknown predictions: `0`

The manuscript also notes that the main confusion pairs are `gneiss <-> schist` and `andesite <-> basalt`, which is consistent with visually similar petrographic textures.

## What the notebook does

The notebook runs an end-to-end fine-tuning pipeline:

1. Detects whether it is running locally or in Google Colab.
2. Installs the required training dependencies.
3. Loads training data from JSON plus an image directory or ZIP archive.
4. Converts samples into the Unsloth vision chat format.
5. Fine-tunes a vision instruction model with LoRA adapters.
6. Runs a quick inference smoke test.
7. Evaluates on a held-out set with classification metrics.
8. Saves the trained LoRA adapter files.

## Relation to the paper

The notebook appears to be the implementation vehicle for the experiment described in the manuscript:

- it uses `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
- it converts caption-style annotations into label-only instruction pairs
- it uses LoRA with rank `32`
- it trains with batch size `2`, gradient accumulation `4`, and `3` epochs
- it evaluates with classification metrics on a held-out set

In other words, the repository is currently a minimal reproduction package centered around the notebook rather than a multi-file Python project.

## Model and training setup

By default, the notebook uses:

- `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
- LoRA fine-tuning over vision and language layers
- 4-bit loading when the model name includes `4bit` or `bnb-4bit`
- `label_only` response mode, which trains the model to return only the rock name
- seed `3407`

The training loop is implemented with:

- `unsloth`
- `transformers`
- `trl`
- `peft`
- `datasets`
- `bitsandbytes`
- `scikit-learn`
- `pandas`
- `pillow`

## Expected dataset format

The notebook expects JSON records shaped like this:

```json
[
  {
    "image": "104_01311.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nClassify this petrographic thin section image. What type of rock is this?"
      },
      {
        "from": "gpt",
        "value": "Diorite. This is a coarse-grained intrusive igneous rock..."
      }
    ]
  }
]
```

Important details:

- The notebook reads `image` and `conversations`.
- In `label_only` mode, only the first label before the first period is used for training.
- The image path is normalized to the file name, then joined with `RX_IMAGES_DIR`.

The manuscript describes a balanced 14-class dataset with:

- `200` plane-polarized-light images per class
- `200` cross-polarized-light images per class
- `400` images per class in total

## Required files

For a typical Colab run, place these in your Drive folder:

- `train_data.json`
- `test_data.json` (optional)
- `RX_BASE_Imagenes.zip`

The ZIP is expected to contain an `Imagenes/` folder with `.jpg` images.

## Configuration

The notebook is controlled with environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RX_DRIVE_FOLDER` | `/content/drive/MyDrive/Internship/RX_BASE_Finetune` | Colab Drive folder containing data |
| `RX_ZIP_FILENAME` | `RX_BASE_Imagenes.zip` | ZIP file containing the image set |
| `RX_IMAGES_DIR` | `/content/images` | Directory used for extracted images |
| `RX_TRAIN_JSON` | `train_data.json` in the Drive folder | Training JSON path |
| `RX_TEST_JSON` | `test_data.json` in the Drive folder | Optional test JSON path |
| `RX_ANSWER_STYLE` | `label_only` | `label_only` or `full` training targets |
| `RX_MODEL` | `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` | Base model |
| `RX_SUBSET` | `0` | Limit dataset size for quick experiments |
| `RX_MAX_STEPS` | `350` | Training steps when epoch mode is disabled |
| `RX_BATCH_SIZE` | `2` | Per-device batch size |
| `RX_GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `RX_EPOCHS` | `3` | If greater than `0`, derives `MAX_STEPS` from epochs |
| `RX_EVAL_SIZE` | `100` | Number of evaluation samples, `0` means all |
| `RX_EVAL_TEMPERATURE` | `0.05` | Generation temperature during evaluation |

## Running the notebook

### Option 1: Google Colab

1. Upload the repository notebook to Colab.
2. Put `train_data.json`, optional `test_data.json`, and `RX_BASE_Imagenes.zip` in your Drive folder.
3. Update any environment variables you want to change.
4. Run the notebook from top to bottom.

The notebook will:

- mount Google Drive
- unzip images into `/content/images`
- train the model
- evaluate it
- save LoRA adapters locally and copy them into `Models/` in Drive

### Option 2: Local Jupyter environment

1. Create a Python environment with GPU-enabled PyTorch.
2. Launch Jupyter.
3. Open `RX-11B-Full-3E.ipynb`.
4. Set local paths through environment variables before starting Jupyter, for example:

```bash
export RX_IMAGES_DIR="/absolute/path/to/images"
export RX_TRAIN_JSON="/absolute/path/to/train_data.json"
export RX_TEST_JSON="/absolute/path/to/test_data.json"
export RX_MODEL="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
```

Then run the notebook cells in order.

## Reproducing the manuscript configuration

To match the reported experiment as closely as possible, use the notebook with settings equivalent to:

```bash
export RX_MODEL="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
export RX_ANSWER_STYLE="label_only"
export RX_BATCH_SIZE="2"
export RX_GRAD_ACCUM="4"
export RX_EPOCHS="3"
export RX_EVAL_TEMPERATURE="0.05"
```

The notebook itself also sets:

- optimizer: `adamw_8bit`
- learning rate: `1e-4`
- scheduler: `cosine`
- weight decay: `0.01`
- seed: `3407`

## Evaluation

After training, the notebook evaluates the model on:

- `test_data.json` if it exists
- otherwise, a deterministic 20% split from `train_data.json`

It reports:

- accuracy
- macro and weighted F1
- macro precision and recall
- per-class classification report
- confusion matrix as a Pandas DataFrame

## Outputs

Training produces:

- intermediate checkpoints in `outputs/`
- final LoRA adapters in a timestamped directory like `rx_lora_YYYYMMDD_HHMMSS/`

If the notebook is running in Colab, the final adapter directory is also copied to:

```text
<RX_DRIVE_FOLDER>/Models/<timestamped_adapter_dir>
```

## Notes and caveats

- Large non-quantized vision models can fail because of CPU offloading and meta tensors.
- The notebook includes a guard that raises an error if model parameters remain on the meta device.
- The repository currently does not include the dataset, helper scripts, or a packaged training CLI; the notebook is the main entry point.
- The manuscript PDF contains additional context, literature comparison, and reported benchmark tables that are not stored directly in the repository.

## Repository contents

```text
.
├── README.md
└── RX-11B-Full-3E.ipynb
```
