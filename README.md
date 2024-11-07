## HFKD


## Requirements
- Python3
- PyTorch (> 1.2.0)
- torchvision
- numpy
- timm  (0.6.4)

## Training
We use the ```.ymal``` file as the configuration file, where the specific hyperparameter configuration information is available.At training time, use the ```config``` parameter to specify the path to the ymal file.
Run ```train.py``` with student network as ResNet18 and teacher as Swin-T  to reproduce experiment result on CIFAR100.
```
python train.py --config ./config/diffkd-cifar100_SwinT--resnet18.ymal
```
Run ```train.py``` with student network as ResMLP and teacher as Swin-T  to reproduce experiment result on CIFAR100.
```
python train.py --config ./config/diffkd-cifar100_SwinT--ResMLP.ymal
```
Run ```train.py``` with student network as ResMLP and teacher as Swin-T  to reproduce experiment result on ImageNet.
```
python train.py --config ./config/diffkd_imagenet_cifar100_SwinT--ResMLP.ymal
```

## License

