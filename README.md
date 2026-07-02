# PromptRad

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