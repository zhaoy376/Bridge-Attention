# Bridge-Attention
The code for the paper 'BA-Net: Bridge Attention for Deep Convolutional Neural Networks'
## Usage
### Training
If you want to train the BA-Net under the backbones like ResNet, ResNeXt, MobileNetv3, use the code ↓. 
```Python
python main.py -a ba_resnet50 {datapath_of_ImageNet} --lr 0.1 --scheduler cos -b 256
```
