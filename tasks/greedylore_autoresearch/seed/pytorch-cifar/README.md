# Train CIFAR10 with PyTorch

## Dense Muon / GreedyLore + Muon paper protocol

`train_ddp.py` reuses the original CIFAR ResNet, augmentation and normalization.
The experiment uses the approved paper settings: 40 epochs, four GPUs, local
batch 32, epoch cosine decay to zero, rank 64, EF14; compression starts after
500/4000 dense steps and bases refresh every 750/1200 steps for CIFAR-10/100.
Matrix LR is 0.02; scalar AdamW LR is 0.005/0.0005. Weight decay is 5e-4 except
for bias and normalization parameters. These Muon settings are adaptations,
not hyperparameters from the paper. No checkpoints are written.

```bash
bash pytorch-cifar/run_paper_muon.bash cifar10 dense --dry-run
bash pytorch-cifar/run_paper_muon.bash cifar100 greedylore --dry-run
# From the repository root, inside tmux:
bash pytorch-cifar/queue_paper_muon.bash cifar10
bash pytorch-cifar/queue_paper_muon.bash cifar100
```

The dataset queues use GPUs 0–3 and 4–7 respectively, run Dense before
GreedyLore, and use separate online W&B projects `GreedyLore-CIFAR10-Muon`
and `GreedyLore-CIFAR100-Muon`. Runs CM044–CM047 save configuration, provenance,
logs, epoch metrics, tracker IDs and final results under `outputs/<run-id>/`.
Test evaluation covers exactly 10,000 images; final and best accuracy are
separate fields. Communication counts estimate per-rank collective inputs,
including dense warmup/refresh steps and score probes, rather than network
traffic; Muon's own collective counters are reported separately.

Reference: [GreedyLore, Figure 3 and Appendix F](https://arxiv.org/pdf/2507.08784).

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |
