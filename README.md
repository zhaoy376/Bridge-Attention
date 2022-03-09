# Bridge-Attention
The code for the paper 'BA-Net: Bridge Attention for Deep Convolutional Neural Networks'
## Usage
### Training
If you want to train the BA-Net under the backbone architectures like ResNet, ResNeXt, MobileNetv3, use the code ↓. 
```Python
python main.py -a ba_resnet50 {datapath_of_ImageNet} --lr 0.1 --scheduler cos -b 256
# -a: the target architecture, including ba_resnet{18/34/50/101/152}, ba_resnext{18/34/50/101/152}, ba_mobilenetv3_large and ba_mobilenetv3_small. 
Or you want to train the origin architectures, we also provide resnext{18/34/50/101/152}, mobilenetv3_large and obilenetv3_small. 
# {datapath_of_ImageNet}: directly type the path of the ImageNet-1K, which should contain directories of 'train' and 'val'.
# --lr: the initial learning rate. 
# --scheduler: the training scheduler, including 'cos' and 'step'. 
# -b: batchsize. 
```
