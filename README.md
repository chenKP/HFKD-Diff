## HFKD


## Requirements
- Python3
- PyTorch (> 1.2.0)
- torchvision
- numpy


## Training
Run ```train.py``` with student network as ResNet18 and teacher as Swin-T  to reproduce experiment result on CIFAR100.
```
python train.py --config ./config/diffkd-cifar100_SwinT--resnet18.ymal
```

## License

# HFKD