# torchtools

## torchtools.utils.print_summary

* Highlights:
  * Calculate FLOPs for **RNN, LSTM, GRU**
  * Calculate FLOPs for **Attention** (in Vision Transformer)

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
---------------------------------------------------------------------------------------------------------------------
                            Layer (type)    Output shape     Param shape      Param #     FLOPs basic           FLOPs
=====================================================================================================================
                                 Input *     1x3x224x224
                        conv1 (Conv2d) *    1x64x112x112        64x3x7x7        9,408     118,013,952     118,013,952
                     bn1 (BatchNorm2d) *    1x64x112x112           64+64          128               0       1,605,632
                           relu (ReLU) *    1x64x112x112                            0               0               0
                   maxpool (MaxPool2d) *      1x64x56x56                            0               0               0
               layer1.0.conv1 (Conv2d) *      1x64x56x56       64x64x3x3       36,864     115,605,504     115,605,504
            layer1.0.bn1 (BatchNorm2d) *      1x64x56x56           64+64          128               0         401,408
                  layer1.0.relu (ReLU) *      1x64x56x56                            0               0               0
               layer1.0.conv2 (Conv2d) *      1x64x56x56       64x64x3x3       36,864     115,605,504     115,605,504
            layer1.0.bn2 (BatchNorm2d) *      1x64x56x56           64+64          128               0         401,408
                  layer1.0.relu (ReLU) *      1x64x56x56                            0               0               0
                 layer1.0 (BasicBlock)        1x64x56x56                            0               0               0
               layer1.1.conv1 (Conv2d) *      1x64x56x56       64x64x3x3       36,864     115,605,504     115,605,504
            layer1.1.bn1 (BatchNorm2d) *      1x64x56x56           64+64          128               0         401,408
                  layer1.1.relu (ReLU) *      1x64x56x56                            0               0               0
               layer1.1.conv2 (Conv2d) *      1x64x56x56       64x64x3x3       36,864     115,605,504     115,605,504
            layer1.1.bn2 (BatchNorm2d) *      1x64x56x56           64+64          128               0         401,408
                  layer1.1.relu (ReLU) *      1x64x56x56                            0               0               0
                 layer1.1 (BasicBlock)        1x64x56x56                            0               0               0
                   layer1 (Sequential)        1x64x56x56                            0               0               0
               layer2.0.conv1 (Conv2d) *     1x128x28x28      128x64x3x3       73,728      57,802,752      57,802,752
            layer2.0.bn1 (BatchNorm2d) *     1x128x28x28         128+128          256               0         200,704
                  layer2.0.relu (ReLU) *     1x128x28x28                            0               0               0
               layer2.0.conv2 (Conv2d) *     1x128x28x28     128x128x3x3      147,456     115,605,504     115,605,504
            layer2.0.bn2 (BatchNorm2d) *     1x128x28x28         128+128          256               0         200,704
        layer2.0.downsample.0 (Conv2d) *     1x128x28x28      128x64x1x1        8,192       6,422,528       6,422,528
   layer2.0.downsample.1 (BatchNorm2d) *     1x128x28x28         128+128          256               0         200,704
      layer2.0.downsample (Sequential)       1x128x28x28                            0               0               0
                  layer2.0.relu (ReLU) *     1x128x28x28                            0               0               0
                 layer2.0 (BasicBlock)       1x128x28x28                            0               0               0
               layer2.1.conv1 (Conv2d) *     1x128x28x28     128x128x3x3      147,456     115,605,504     115,605,504
            layer2.1.bn1 (BatchNorm2d) *     1x128x28x28         128+128          256               0         200,704
                  layer2.1.relu (ReLU) *     1x128x28x28                            0               0               0
               layer2.1.conv2 (Conv2d) *     1x128x28x28     128x128x3x3      147,456     115,605,504     115,605,504
            layer2.1.bn2 (BatchNorm2d) *     1x128x28x28         128+128          256               0         200,704
                  layer2.1.relu (ReLU) *     1x128x28x28                            0               0               0
                 layer2.1 (BasicBlock)       1x128x28x28                            0               0               0
                   layer2 (Sequential)       1x128x28x28                            0               0               0
               layer3.0.conv1 (Conv2d) *     1x256x14x14     256x128x3x3      294,912      57,802,752      57,802,752
            layer3.0.bn1 (BatchNorm2d) *     1x256x14x14         256+256          512               0         100,352
                  layer3.0.relu (ReLU) *     1x256x14x14                            0               0               0
               layer3.0.conv2 (Conv2d) *     1x256x14x14     256x256x3x3      589,824     115,605,504     115,605,504
            layer3.0.bn2 (BatchNorm2d) *     1x256x14x14         256+256          512               0         100,352
        layer3.0.downsample.0 (Conv2d) *     1x256x14x14     256x128x1x1       32,768       6,422,528       6,422,528
   layer3.0.downsample.1 (BatchNorm2d) *     1x256x14x14         256+256          512               0         100,352
      layer3.0.downsample (Sequential)       1x256x14x14                            0               0               0
                  layer3.0.relu (ReLU) *     1x256x14x14                            0               0               0
                 layer3.0 (BasicBlock)       1x256x14x14                            0               0               0
               layer3.1.conv1 (Conv2d) *     1x256x14x14     256x256x3x3      589,824     115,605,504     115,605,504
            layer3.1.bn1 (BatchNorm2d) *     1x256x14x14         256+256          512               0         100,352
                  layer3.1.relu (ReLU) *     1x256x14x14                            0               0               0
               layer3.1.conv2 (Conv2d) *     1x256x14x14     256x256x3x3      589,824     115,605,504     115,605,504
            layer3.1.bn2 (BatchNorm2d) *     1x256x14x14         256+256          512               0         100,352
                  layer3.1.relu (ReLU) *     1x256x14x14                            0               0               0
                 layer3.1 (BasicBlock)       1x256x14x14                            0               0               0
                   layer3 (Sequential)       1x256x14x14                            0               0               0
               layer4.0.conv1 (Conv2d) *       1x512x7x7     512x256x3x3    1,179,648      57,802,752      57,802,752
            layer4.0.bn1 (BatchNorm2d) *       1x512x7x7         512+512        1,024               0          50,176
                  layer4.0.relu (ReLU) *       1x512x7x7                            0               0               0
               layer4.0.conv2 (Conv2d) *       1x512x7x7     512x512x3x3    2,359,296     115,605,504     115,605,504
            layer4.0.bn2 (BatchNorm2d) *       1x512x7x7         512+512        1,024               0          50,176
        layer4.0.downsample.0 (Conv2d) *       1x512x7x7     512x256x1x1      131,072       6,422,528       6,422,528
   layer4.0.downsample.1 (BatchNorm2d) *       1x512x7x7         512+512        1,024               0          50,176
      layer4.0.downsample (Sequential)         1x512x7x7                            0               0               0
                  layer4.0.relu (ReLU) *       1x512x7x7                            0               0               0
                 layer4.0 (BasicBlock)         1x512x7x7                            0               0               0
               layer4.1.conv1 (Conv2d) *       1x512x7x7     512x512x3x3    2,359,296     115,605,504     115,605,504
            layer4.1.bn1 (BatchNorm2d) *       1x512x7x7         512+512        1,024               0          50,176
                  layer4.1.relu (ReLU) *       1x512x7x7                            0               0               0
               layer4.1.conv2 (Conv2d) *       1x512x7x7     512x512x3x3    2,359,296     115,605,504     115,605,504
            layer4.1.bn2 (BatchNorm2d) *       1x512x7x7         512+512        1,024               0          50,176
                  layer4.1.relu (ReLU) *       1x512x7x7                            0               0               0
                 layer4.1 (BasicBlock)         1x512x7x7                            0               0               0
                   layer4 (Sequential)         1x512x7x7                            0               0               0
           avgpool (AdaptiveAvgPool2d) *       1x512x1x1                            0               0               0
                           fc (Linear) *          1x1000   1000x512+1000      513,000         512,000         512,500
                              (ResNet)            1x1000                            0               0               0
---------------------------------------------------------------------------------------------------------------------
Total params: 11,689,512 (44.591949462890625 MB)
Total params (with aux): 11,689,512 (44.591949462890625 MB)
    Trainable params: 11,689,512 (44.591949462890625 MB)
    Non-trainable params: 0 (0.0 MB)
Total flops (basic): 1,814,073,344 (1.814073344 billion)
Total flops: 1,819,041,268 (1.819041268 billion)
---------------------------------------------------------------------------------------------------------------------
NOTE:
    *: leaf modules
    Flops is measured in multiply-adds. Multiply, add, divide, exp are treated the same for calculation (1/2 multiply-adds).
    Flops (basic) only calculates for convolution and linear layers (not inlcude bias)
    Flops additionally calculates for bias, normalization (BatchNorm, LayerNorm, GroupNorm), RNN (RNN, LSTM, GRU) and attention layers
        - activations (e.g. ReLU), operations implemented as functionals (e.g. add in a residual architecture) are not 
          calculated as they are usually neglectable.
        - complex custom module may need manual calculation for correctness (refer to RNN, LSTM, GRU, Attention as examples).
---------------------------------------------------------------------------------------------------------------------
Out[1]: 
{'flops': 1819041268,
 'flops_basic': 1814073344,
 'params': 11689512,
 'params_with_aux': 11689512}
```

```python
import torch
import timm.models as models
from torchtools.utils import print_summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.vit_base_patch16_224().to(device)
inputs = torch.randn(1, 3, 224, 224).to(device)

print_summary(model, inputs)
```

```
---------------------------------------------------------------------------------------------------------------------
                            Layer (type)    Output shape     Param shape      Param #     FLOPs basic           FLOPs
=====================================================================================================================
                                 Input *     1x3x224x224
             patch_embed.proj (Conv2d) *     1x768x14x14 768x3x16x16+768      590,592     115,605,504     115,680,768
           patch_embed.norm (Identity) *       1x196x768                            0               0               0
              patch_embed (PatchEmbed)         1x196x768                            0               0               0
                    pos_drop (Dropout) *       1x197x768                            0               0               0
            blocks.0.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.0.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.0.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.0.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.0.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.0.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.0.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.0.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.0.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.0.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.0.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.0.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.0.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.0.mlp (Mlp)         1x197x768                            0               0               0
         blocks.0.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.0 (Block)         1x197x768                            0               0               0
            blocks.1.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.1.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.1.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.1.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.1.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.1.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.1.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.1.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.1.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.1.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.1.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.1.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.1.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.1.mlp (Mlp)         1x197x768                            0               0               0
         blocks.1.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.1 (Block)         1x197x768                            0               0               0
            blocks.2.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.2.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.2.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.2.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.2.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.2.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.2.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.2.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.2.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.2.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.2.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.2.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.2.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.2.mlp (Mlp)         1x197x768                            0               0               0
         blocks.2.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.2 (Block)         1x197x768                            0               0               0
            blocks.3.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.3.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.3.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.3.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.3.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.3.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.3.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.3.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.3.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.3.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.3.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.3.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.3.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.3.mlp (Mlp)         1x197x768                            0               0               0
         blocks.3.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.3 (Block)         1x197x768                            0               0               0
            blocks.4.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.4.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.4.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.4.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.4.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.4.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.4.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.4.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.4.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.4.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.4.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.4.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.4.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.4.mlp (Mlp)         1x197x768                            0               0               0
         blocks.4.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.4 (Block)         1x197x768                            0               0               0
            blocks.5.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.5.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.5.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.5.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.5.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.5.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.5.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.5.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.5.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.5.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.5.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.5.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.5.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.5.mlp (Mlp)         1x197x768                            0               0               0
         blocks.5.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.5 (Block)         1x197x768                            0               0               0
            blocks.6.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.6.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.6.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.6.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.6.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.6.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.6.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.6.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.6.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.6.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.6.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.6.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.6.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.6.mlp (Mlp)         1x197x768                            0               0               0
         blocks.6.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.6 (Block)         1x197x768                            0               0               0
            blocks.7.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.7.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.7.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.7.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.7.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.7.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.7.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.7.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.7.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.7.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.7.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.7.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.7.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.7.mlp (Mlp)         1x197x768                            0               0               0
         blocks.7.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.7 (Block)         1x197x768                            0               0               0
            blocks.8.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.8.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.8.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.8.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.8.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.8.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.8.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.8.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.8.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.8.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.8.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.8.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.8.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.8.mlp (Mlp)         1x197x768                            0               0               0
         blocks.8.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.8 (Block)         1x197x768                            0               0               0
            blocks.9.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.9.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
     blocks.9.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
           blocks.9.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
     blocks.9.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
             blocks.9.attn (Attention)         1x197x768                            0               0     179,285,760
         blocks.9.drop_path (Identity) *       1x197x768                            0               0               0
            blocks.9.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
             blocks.9.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
               blocks.9.mlp.act (GELU) *      1x197x3072                            0               0               0
           blocks.9.mlp.drop (Dropout) *      1x197x3072                            0               0               0
             blocks.9.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
           blocks.9.mlp.drop (Dropout) *       1x197x768                            0               0               0
                    blocks.9.mlp (Mlp)         1x197x768                            0               0               0
         blocks.9.drop_path (Identity) *       1x197x768                            0               0               0
                      blocks.9 (Block)         1x197x768                            0               0               0
           blocks.10.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
           blocks.10.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
    blocks.10.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
          blocks.10.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
    blocks.10.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
            blocks.10.attn (Attention)         1x197x768                            0               0     179,285,760
        blocks.10.drop_path (Identity) *       1x197x768                            0               0               0
           blocks.10.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.10.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
              blocks.10.mlp.act (GELU) *      1x197x3072                            0               0               0
          blocks.10.mlp.drop (Dropout) *      1x197x3072                            0               0               0
            blocks.10.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
          blocks.10.mlp.drop (Dropout) *       1x197x768                            0               0               0
                   blocks.10.mlp (Mlp)         1x197x768                            0               0               0
        blocks.10.drop_path (Identity) *       1x197x768                            0               0               0
                     blocks.10 (Block)         1x197x768                            0               0               0
           blocks.11.norm1 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
           blocks.11.attn.qkv (Linear) *      1x197x2304   2304x768+2304    1,771,776     348,585,984     348,812,928
    blocks.11.attn.attn_drop (Dropout) *    1x12x197x197                            0               0               0
          blocks.11.attn.proj (Linear) *       1x197x768     768x768+768      590,592     116,195,328     116,270,976
    blocks.11.attn.proj_drop (Dropout) *       1x197x768                            0               0               0
            blocks.11.attn (Attention)         1x197x768                            0               0     179,285,760
        blocks.11.drop_path (Identity) *       1x197x768                            0               0               0
           blocks.11.norm2 (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
            blocks.11.mlp.fc1 (Linear) *      1x197x3072   3072x768+3072    2,362,368     464,781,312     465,083,904
              blocks.11.mlp.act (GELU) *      1x197x3072                            0               0               0
          blocks.11.mlp.drop (Dropout) *      1x197x3072                            0               0               0
            blocks.11.mlp.fc2 (Linear) *       1x197x768    768x3072+768    2,360,064     464,781,312     464,856,960
          blocks.11.mlp.drop (Dropout) *       1x197x768                            0               0               0
                   blocks.11.mlp (Mlp)         1x197x768                            0               0               0
        blocks.11.drop_path (Identity) *       1x197x768                            0               0               0
                     blocks.11 (Block)         1x197x768                            0               0               0
                   blocks (Sequential)         1x197x768                            0               0               0
                      norm (LayerNorm) *       1x197x768         768+768        1,536               0         302,592
                 pre_logits (Identity) *           1x768                            0               0               0
                         head (Linear) *          1x1000   1000x768+1000      769,000         768,000         768,500
                   (VisionTransformer)            1x1000 1x1x768+1x197x768            0               0               0
---------------------------------------------------------------------------------------------------------------------
Total params: 86,415,592 (329.6493225097656 MB)
Total params (with aux): 86,567,656 (330.2294006347656 MB)
    Trainable params: 86,567,656 (330.2294006347656 MB)
    Non-trainable params: 0 (0.0 MB)
Total flops (basic): 16,848,500,736 (16.848500736 billion)
Total flops: 19,015,740,404 (19.015740404 billion)
---------------------------------------------------------------------------------------------------------------------
NOTE:
    *: leaf modules
    Flops is measured in multiply-adds. Multiply, add, divide, exp are treated the same for calculation (1/2 multiply-adds).
    Flops (basic) only calculates for convolution and linear layers (not inlcude bias)
    Flops additionally calculates for bias, normalization (BatchNorm, LayerNorm, GroupNorm), RNN (RNN, LSTM, GRU) and attention layers
        - activations (e.g. ReLU), operations implemented as functionals (e.g. add in a residual architecture) are not 
          calculated as they are usually neglectable.
        - complex custom module may need manual calculation for correctness (refer to RNN, LSTM, GRU, Attention as examples).
---------------------------------------------------------------------------------------------------------------------
Out[2]: 
{'flops': 19015740404,
 'flops_basic': 16848500736,
 'params': 86415592,
 'params_with_aux': 86567656}
```
