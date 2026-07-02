# PromptRad

Prompt-based multi-label classification of liver radiology reports.
This repository provides **inference code and pre-trained model weights** so that
you can reproduce the reported results. (The training data is not publicly
released, so training code is not included.)

Two model variants are available:

| Variant | HF model | Description |
|---|---|---|
| `promptrad` | [`ilalab/promptrad`](https://huggingface.co/ilalab/promptrad) | Manual prompt template |
| `promptrad-autot` | [`ilalab/promptrad-autot`](https://huggingface.co/ilalab/promptrad-autot) | Automatically generated prompt template |

Each model is released as **5 seeds** (`seed_0, 1, 3, 7, 10`); the reported
numbers are the mean ± std over these seeds.

## Installation
### Create a virtual environment (we use [`uv`](https://github.com/astral-sh/uv))
```bash
uv venv --python=3.9
source .venv/bin/activate
uv init
```
### Install dependencies
```bash
cat requirements.txt | xargs -n 1 uv add
uv pip install torch==1.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

> **GPU note.** The command above installs a CUDA 10.2 build of PyTorch, which
> supports GPUs up to `sm_70`. If you have a newer GPU (e.g. RTX 30/40 series,
> which are `sm_86`+), install a matching build instead, for example:
> ```bash
> uv pip install 'torch==1.10.0+cu111' -f https://download.pytorch.org/whl/cu111/torch_stable.html
> ```
> You can check which architectures your install supports with
> `python -c "import torch; print(torch.cuda.get_arch_list())"`.
> Inference also runs fine on CPU (prepend `CUDA_VISIBLE_DEVICES=""`).

## Download the models
Download the checkpoints into `checkpoints/<variant>/` with the Hugging Face CLI:

```bash
hf download ilalab/promptrad        --local-dir checkpoints/promptrad
hf download ilalab/promptrad-autot  --local-dir checkpoints/promptrad-autot
```

This produces:
```
checkpoints/
├── promptrad/
│   ├── seed_0/   (config.json, pytorch_model.bin, tokenizer files, args.json)
│   ├── seed_1/
│   ├── seed_3/
│   ├── seed_7/
│   └── seed_10/
└── promptrad-autot/
    └── seed_{0,1,3,7,10}/
```

## Prepare the data
Place two files under `data/`:

- **`data/test.pkl`** — a pickled `pandas.DataFrame` with the columns:
  | column | type | description |
  |---|---|---|
  | `doc_idx` | int | document id |
  | `X` | str | the radiology report text |
  | `y_true` | list[int] | 7 binary labels, ordered as in `class_names.pkl` |

- **`data/class_names.pkl`** — a pickled `dict` mapping each class name to its
  index. The label order must match the `y_true` vectors, e.g.:
  ```python
  {'cyst': 0, 'HCC': 1, 'post-treatment': 2, 'cirrhosis': 3,
   'steatosis': 4, 'metastasis': 5, 'hemangioma': 6}
  ```

## Run inference
Run from the repository root. Predictions for all 5 seeds are written to
`results/train_32/<variant>/seed_<seed>/predict/predictions.pkl`.

```bash
python src/test.py --test_method promptrad        --test_filename test.pkl
python src/test.py --test_method promptrad-autot  --test_filename test.pkl
```

## Evaluate
Print the per-class / macro / micro F1 averaged over the 5 seeds:

```bash
python src/evaluate.py --test_method promptrad        --test_filename test.pkl
python src/evaluate.py --test_method promptrad-autot  --test_filename test.pkl
```
