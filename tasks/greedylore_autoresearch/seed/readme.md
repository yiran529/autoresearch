### set up environment

```shell
conda create -n ccb python=3.11 -y
conda activate ccb
# for cuda-118
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# for cuda-121
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# for cuda-124
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

we use cuda_12.1

### requirements

```
transformers
accelerate
datasets
evaluate
wandb
loguru
cupy
```

