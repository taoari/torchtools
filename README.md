# torchtools

## torchtools.utils.print_summary


```python
import torch
import torchvision.models as models
from torchtools.utils import print_summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.resnet18().to(device)
inputs = torch.randn(1, 3, 224, 224).to(device)

print_summary(model, inputs)
```

```
Model summary:
--------------------------------------------------------------------------------------------------
                            Layer (type)    Output shape      Param #           Flops   Memory (B)
==================================================================================================
                                   Input     1x3x224x224
                          conv1 (Conv2d)    1x64x112x112        9,408     118,013,952    3,211,264
                       bn1 (BatchNorm2d)    1x64x112x112          128               0    3,211,264
                             relu (ReLU)    1x64x112x112            0               0            0
                     maxpool (MaxPool2d)      1x64x56x56            0               0      802,816
                 layer1.0.conv1 (Conv2d)      1x64x56x56       36,864     115,605,504      802,816
              layer1.0.bn1 (BatchNorm2d)      1x64x56x56          128               0      802,816
                    layer1.0.relu (ReLU)      1x64x56x56            0               0            0
                 layer1.0.conv2 (Conv2d)      1x64x56x56       36,864     115,605,504      802,816
              layer1.0.bn2 (BatchNorm2d)      1x64x56x56          128               0      802,816
                    layer1.0.relu (ReLU)      1x64x56x56            0               0            0
                 layer1.1.conv1 (Conv2d)      1x64x56x56       36,864     115,605,504      802,816
              layer1.1.bn1 (BatchNorm2d)      1x64x56x56          128               0      802,816
                    layer1.1.relu (ReLU)      1x64x56x56            0               0            0
                 layer1.1.conv2 (Conv2d)      1x64x56x56       36,864     115,605,504      802,816
              layer1.1.bn2 (BatchNorm2d)      1x64x56x56          128               0      802,816
                    layer1.1.relu (ReLU)      1x64x56x56            0               0            0
                 layer2.0.conv1 (Conv2d)     1x128x28x28       73,728      57,802,752      401,408
              layer2.0.bn1 (BatchNorm2d)     1x128x28x28          256               0      401,408
                    layer2.0.relu (ReLU)     1x128x28x28            0               0            0
                 layer2.0.conv2 (Conv2d)     1x128x28x28      147,456     115,605,504      401,408
              layer2.0.bn2 (BatchNorm2d)     1x128x28x28          256               0      401,408
          layer2.0.downsample.0 (Conv2d)     1x128x28x28        8,192       6,422,528      401,408
     layer2.0.downsample.1 (BatchNorm2d)     1x128x28x28          256               0      401,408
                    layer2.0.relu (ReLU)     1x128x28x28            0               0            0
                 layer2.1.conv1 (Conv2d)     1x128x28x28      147,456     115,605,504      401,408
              layer2.1.bn1 (BatchNorm2d)     1x128x28x28          256               0      401,408
                    layer2.1.relu (ReLU)     1x128x28x28            0               0            0
                 layer2.1.conv2 (Conv2d)     1x128x28x28      147,456     115,605,504      401,408
              layer2.1.bn2 (BatchNorm2d)     1x128x28x28          256               0      401,408
                    layer2.1.relu (ReLU)     1x128x28x28            0               0            0
                 layer3.0.conv1 (Conv2d)     1x256x14x14      294,912      57,802,752      200,704
              layer3.0.bn1 (BatchNorm2d)     1x256x14x14          512               0      200,704
                    layer3.0.relu (ReLU)     1x256x14x14            0               0            0
                 layer3.0.conv2 (Conv2d)     1x256x14x14      589,824     115,605,504      200,704
              layer3.0.bn2 (BatchNorm2d)     1x256x14x14          512               0      200,704
          layer3.0.downsample.0 (Conv2d)     1x256x14x14       32,768       6,422,528      200,704
     layer3.0.downsample.1 (BatchNorm2d)     1x256x14x14          512               0      200,704
                    layer3.0.relu (ReLU)     1x256x14x14            0               0            0
                 layer3.1.conv1 (Conv2d)     1x256x14x14      589,824     115,605,504      200,704
              layer3.1.bn1 (BatchNorm2d)     1x256x14x14          512               0      200,704
                    layer3.1.relu (ReLU)     1x256x14x14            0               0            0
                 layer3.1.conv2 (Conv2d)     1x256x14x14      589,824     115,605,504      200,704
              layer3.1.bn2 (BatchNorm2d)     1x256x14x14          512               0      200,704
                    layer3.1.relu (ReLU)     1x256x14x14            0               0            0
                 layer4.0.conv1 (Conv2d)       1x512x7x7    1,179,648      57,802,752      100,352
              layer4.0.bn1 (BatchNorm2d)       1x512x7x7        1,024               0      100,352
                    layer4.0.relu (ReLU)       1x512x7x7            0               0            0
                 layer4.0.conv2 (Conv2d)       1x512x7x7    2,359,296     115,605,504      100,352
              layer4.0.bn2 (BatchNorm2d)       1x512x7x7        1,024               0      100,352
          layer4.0.downsample.0 (Conv2d)       1x512x7x7      131,072       6,422,528      100,352
     layer4.0.downsample.1 (BatchNorm2d)       1x512x7x7        1,024               0      100,352
                    layer4.0.relu (ReLU)       1x512x7x7            0               0            0
                 layer4.1.conv1 (Conv2d)       1x512x7x7    2,359,296     115,605,504      100,352
              layer4.1.bn1 (BatchNorm2d)       1x512x7x7        1,024               0      100,352
                    layer4.1.relu (ReLU)       1x512x7x7            0               0            0
                 layer4.1.conv2 (Conv2d)       1x512x7x7    2,359,296     115,605,504      100,352
              layer4.1.bn2 (BatchNorm2d)       1x512x7x7        1,024               0      100,352
                    layer4.1.relu (ReLU)       1x512x7x7            0               0            0
             avgpool (AdaptiveAvgPool2d)       1x512x1x1            0               0        2,048
                             fc (Linear)          1x1000      513,000         512,000        4,000
--------------------------------------------------------------------------------------------------
Total params: 11,689,512 (44.591949462890625 MB)
Total params (with aux): 11,689,512 (44.591949462890625 MB)
    Trainable params: 11,689,512 (44.591949462890625 MB)
    Non-trainable params: 0 (0.0 MB)
Total flops: 1,814,073,344 (1.814073344 billion)
--------------------------------------------------------------------------------------------------
Out[6]: {'flops': 1814073344, 'params': 11689512, 'params_with_aux': 11689512}
```
