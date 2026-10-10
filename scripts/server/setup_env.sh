#!/bin/bash
# One-time environment and asset setup for the DCEF/CAB campaigns (isolated venv).
set -euo pipefail
cd /data/home/202412012250/bishe_dcef
export HF_HOME=/data/home/202412012250/bishe_dcef/.hf
export HF_ENDPOINT=https://hf-mirror.com
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
echo "[$(date)] create venv"
[ -d .venv ] || python3 -m venv --without-pip .venv
if [ ! -x .venv/bin/pip ]; then
  # ensurepip is absent on this host: bootstrap pip into the venv with the user-site pip
  PYTHONPATH=/data/home/202412012250/.local/lib/python3.10/site-packages .venv/bin/python -m pip install -q -I pip -i $MIRROR
fi
. .venv/bin/activate
python -m pip --version
echo "[$(date)] install torch 2.8.0+cu128"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo "[$(date)] install transformers/datasets"
python -m pip install "transformers==4.56.2" "datasets==3.6.0" "numpy>=1.26,<3" "PyYAML>=6.0" "safetensors>=0.4.5" "tokenizers>=0.21.0" pytest -i $MIRROR
python -c "import torch,transformers,datasets;print(\"VERSIONS\",torch.__version__,transformers.__version__,datasets.__version__,\"cuda\",torch.cuda.is_available())"
echo "[$(date)] copy roberta-base"
mkdir -p models/roberta-base
cp -L /data/home/202412012250/.cache/huggingface/hub/models--roberta-base/snapshots/*/* models/roberta-base/
ls -la models/roberta-base
echo "[$(date)] download GLUE"
python - <<"PY"
from datasets import load_dataset
for task in ["sst2", "rte", "mnli"]:
    d = load_dataset("nyu-mll/glue", task)
    d.save_to_disk(f"datasets/{task}")
    print("GLUE", task, {k: len(v) for k, v in d.items()}, d["train"].features["label"].names, flush=True)
PY
echo "[$(date)] SETUP_DONE"
