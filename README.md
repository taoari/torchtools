# torchtools

## torchtools.utils.print_summary

* **Flops** (base) only calculates for convolution and linear layers (not inlcude bias)
* **Flops (full)** additionally calculates for bias, normalization (BatchNorm, LayerNorm, GroupNorm), and **attention** layers
  * activations (e.g. ReLU), operations implemented as functionals are not calculated
  * Flops calcuation for non-standard layers (e.g. attention) can be complex, please refer to `hooks._flops_full` for details

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
------------------------------------------------------------------------------------------------------------------
                            Layer (type)    Output shape      Param #           FLOPs      FLOPs full   Memory (B)
==================================================================================================================
                                 Input *     1x3x224x224
                        conv1 (Conv2d) *    1x64x112x112        9,408     118,013,952     118,013,952    3,211,264
                     bn1 (BatchNorm2d) *    1x64x112x112          128               0       1,605,632    3,211,264
                           relu (ReLU) *    1x64x112x112            0               0               0            0
                   maxpool (MaxPool2d) *      1x64x56x56            0               0               0      802,816
               layer1.0.conv1 (Conv2d) *      1x64x56x56       36,864     115,605,504     115,605,504      802,816
            layer1.0.bn1 (BatchNorm2d) *      1x64x56x56          128               0         401,408      802,816
                  layer1.0.relu (ReLU) *      1x64x56x56            0               0               0            0
               layer1.0.conv2 (Conv2d) *      1x64x56x56       36,864     115,605,504     115,605,504      802,816
            layer1.0.bn2 (BatchNorm2d) *      1x64x56x56          128               0         401,408      802,816
                  layer1.0.relu (ReLU) *      1x64x56x56            0               0               0            0
                 layer1.0 (BasicBlock)        1x64x56x56            0               0               0      802,816
               layer1.1.conv1 (Conv2d) *      1x64x56x56       36,864     115,605,504     115,605,504      802,816
            layer1.1.bn1 (BatchNorm2d) *      1x64x56x56          128               0         401,408      802,816
                  layer1.1.relu (ReLU) *      1x64x56x56            0               0               0            0
               layer1.1.conv2 (Conv2d) *      1x64x56x56       36,864     115,605,504     115,605,504      802,816
            layer1.1.bn2 (BatchNorm2d) *      1x64x56x56          128               0         401,408      802,816
                  layer1.1.relu (ReLU) *      1x64x56x56            0               0               0            0
                 layer1.1 (BasicBlock)        1x64x56x56            0               0               0      802,816
                   layer1 (Sequential)        1x64x56x56            0               0               0      802,816
               layer2.0.conv1 (Conv2d) *     1x128x28x28       73,728      57,802,752      57,802,752      401,408
            layer2.0.bn1 (BatchNorm2d) *     1x128x28x28          256               0         200,704      401,408
                  layer2.0.relu (ReLU) *     1x128x28x28            0               0               0            0
               layer2.0.conv2 (Conv2d) *     1x128x28x28      147,456     115,605,504     115,605,504      401,408
            layer2.0.bn2 (BatchNorm2d) *     1x128x28x28          256               0         200,704      401,408
        layer2.0.downsample.0 (Conv2d) *     1x128x28x28        8,192       6,422,528       6,422,528      401,408
   layer2.0.downsample.1 (BatchNorm2d) *     1x128x28x28          256               0         200,704      401,408
      layer2.0.downsample (Sequential)       1x128x28x28            0               0               0      401,408
                  layer2.0.relu (ReLU) *     1x128x28x28            0               0               0            0
                 layer2.0 (BasicBlock)       1x128x28x28            0               0               0      401,408
               layer2.1.conv1 (Conv2d) *     1x128x28x28      147,456     115,605,504     115,605,504      401,408
            layer2.1.bn1 (BatchNorm2d) *     1x128x28x28          256               0         200,704      401,408
                  layer2.1.relu (ReLU) *     1x128x28x28            0               0               0            0
               layer2.1.conv2 (Conv2d) *     1x128x28x28      147,456     115,605,504     115,605,504      401,408
            layer2.1.bn2 (BatchNorm2d) *     1x128x28x28          256               0         200,704      401,408
                  layer2.1.relu (ReLU) *     1x128x28x28            0               0               0            0
                 layer2.1 (BasicBlock)       1x128x28x28            0               0               0      401,408
                   layer2 (Sequential)       1x128x28x28            0               0               0      401,408
               layer3.0.conv1 (Conv2d) *     1x256x14x14      294,912      57,802,752      57,802,752      200,704
            layer3.0.bn1 (BatchNorm2d) *     1x256x14x14          512               0         100,352      200,704
                  layer3.0.relu (ReLU) *     1x256x14x14            0               0               0            0
               layer3.0.conv2 (Conv2d) *     1x256x14x14      589,824     115,605,504     115,605,504      200,704
            layer3.0.bn2 (BatchNorm2d) *     1x256x14x14          512               0         100,352      200,704
        layer3.0.downsample.0 (Conv2d) *     1x256x14x14       32,768       6,422,528       6,422,528      200,704
   layer3.0.downsample.1 (BatchNorm2d) *     1x256x14x14          512               0         100,352      200,704
      layer3.0.downsample (Sequential)       1x256x14x14            0               0               0      200,704
                  layer3.0.relu (ReLU) *     1x256x14x14            0               0               0            0
                 layer3.0 (BasicBlock)       1x256x14x14            0               0               0      200,704
               layer3.1.conv1 (Conv2d) *     1x256x14x14      589,824     115,605,504     115,605,504      200,704
            layer3.1.bn1 (BatchNorm2d) *     1x256x14x14          512               0         100,352      200,704
                  layer3.1.relu (ReLU) *     1x256x14x14            0               0               0            0
               layer3.1.conv2 (Conv2d) *     1x256x14x14      589,824     115,605,504     115,605,504      200,704
            layer3.1.bn2 (BatchNorm2d) *     1x256x14x14          512               0         100,352      200,704
                  layer3.1.relu (ReLU) *     1x256x14x14            0               0               0            0
                 layer3.1 (BasicBlock)       1x256x14x14            0               0               0      200,704
                   layer3 (Sequential)       1x256x14x14            0               0               0      200,704
               layer4.0.conv1 (Conv2d) *       1x512x7x7    1,179,648      57,802,752      57,802,752      100,352
            layer4.0.bn1 (BatchNorm2d) *       1x512x7x7        1,024               0          50,176      100,352
                  layer4.0.relu (ReLU) *       1x512x7x7            0               0               0            0
               layer4.0.conv2 (Conv2d) *       1x512x7x7    2,359,296     115,605,504     115,605,504      100,352
            layer4.0.bn2 (BatchNorm2d) *       1x512x7x7        1,024               0          50,176      100,352
        layer4.0.downsample.0 (Conv2d) *       1x512x7x7      131,072       6,422,528       6,422,528      100,352
   layer4.0.downsample.1 (BatchNorm2d) *       1x512x7x7        1,024               0          50,176      100,352
      layer4.0.downsample (Sequential)         1x512x7x7            0               0               0      100,352
                  layer4.0.relu (ReLU) *       1x512x7x7            0               0               0            0
                 layer4.0 (BasicBlock)         1x512x7x7            0               0               0      100,352
               layer4.1.conv1 (Conv2d) *       1x512x7x7    2,359,296     115,605,504     115,605,504      100,352
            layer4.1.bn1 (BatchNorm2d) *       1x512x7x7        1,024               0          50,176      100,352
                  layer4.1.relu (ReLU) *       1x512x7x7            0               0               0            0
               layer4.1.conv2 (Conv2d) *       1x512x7x7    2,359,296     115,605,504     115,605,504      100,352
            layer4.1.bn2 (BatchNorm2d) *       1x512x7x7        1,024               0          50,176      100,352
                  layer4.1.relu (ReLU) *       1x512x7x7            0               0               0            0
                 layer4.1 (BasicBlock)         1x512x7x7            0               0               0      100,352
                   layer4 (Sequential)         1x512x7x7            0               0               0      100,352
           avgpool (AdaptiveAvgPool2d) *       1x512x1x1            0               0               0        2,048
                           fc (Linear) *          1x1000      513,000         512,000         512,500        4,000
                              (ResNet)            1x1000            0               0               0        4,000
------------------------------------------------------------------------------------------------------------------
Total params: 11,689,512 (44.591949462890625 MB)
Total params (with aux): 11,689,512 (44.591949462890625 MB)
    Trainable params: 11,689,512 (44.591949462890625 MB)
    Non-trainable params: 0 (0.0 MB)
Total flops: 1,814,073,344 (1.814073344 billion)
Total flops (full): 1,819,041,268 (1.819041268 billion)
------------------------------------------------------------------------------------------------------------------
NOTE:
    *: leaf modules
    Flops is measured in multiply-adds, and it only calculates for convolution and linear layers (not inlcude bias)
    Flops (full) additionally calculates for bias, normalization (BatchNorm, LayerNorm, GroupNorm), and attention layers
        - multiply, add, divide, exp are treated the same for calculation (1/2 multiply-adds).
        - activations (e.g. ReLU), operations implemented as functionals (e.g. add in a residual architecture) are not 
          calculated as they are usually neglectable.
------------------------------------------------------------------------------------------------------------------
Out[1]: 
{'flops': 1814073344,
 'flops_full': 1819041268,
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
------------------------------------------------------------------------------------------------------------------
                            Layer (type)    Output shape      Param #           FLOPs      FLOPs full   Memory (B)
==================================================================================================================
                                 Input *     1x3x224x224
             patch_embed.proj (Conv2d) *     1x768x14x14      590,592     115,605,504     115,680,768      602,112
           patch_embed.norm (Identity) *       1x196x768            0               0               0      602,112
              patch_embed (PatchEmbed)         1x196x768            0               0               0      602,112
                    pos_drop (Dropout) *       1x197x768            0               0               0      605,184
            blocks.0.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.0.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.0.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.0.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.0.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.0.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.0.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.0.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.0.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.0.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.0.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.0.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.0.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.0.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.0.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.0 (Block)         1x197x768            0               0               0      605,184
            blocks.1.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.1.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.1.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.1.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.1.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.1.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.1.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.1.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.1.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.1.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.1.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.1.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.1.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.1.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.1.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.1 (Block)         1x197x768            0               0               0      605,184
            blocks.2.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.2.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.2.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.2.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.2.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.2.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.2.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.2.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.2.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.2.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.2.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.2.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.2.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.2.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.2.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.2 (Block)         1x197x768            0               0               0      605,184
            blocks.3.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.3.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.3.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.3.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.3.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.3.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.3.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.3.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.3.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.3.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.3.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.3.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.3.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.3.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.3.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.3 (Block)         1x197x768            0               0               0      605,184
            blocks.4.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.4.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.4.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.4.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.4.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.4.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.4.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.4.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.4.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.4.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.4.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.4.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.4.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.4.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.4.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.4 (Block)         1x197x768            0               0               0      605,184
            blocks.5.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.5.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.5.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.5.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.5.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.5.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.5.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.5.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.5.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.5.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.5.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.5.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.5.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.5.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.5.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.5 (Block)         1x197x768            0               0               0      605,184
            blocks.6.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.6.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.6.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.6.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.6.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.6.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.6.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.6.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.6.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.6.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.6.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.6.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.6.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.6.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.6.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.6 (Block)         1x197x768            0               0               0      605,184
            blocks.7.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.7.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.7.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.7.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.7.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.7.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.7.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.7.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.7.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.7.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.7.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.7.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.7.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.7.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.7.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.7 (Block)         1x197x768            0               0               0      605,184
            blocks.8.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.8.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.8.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.8.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.8.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.8.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.8.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.8.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.8.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.8.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.8.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.8.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.8.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.8.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.8.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.8 (Block)         1x197x768            0               0               0      605,184
            blocks.9.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.9.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
     blocks.9.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
           blocks.9.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
     blocks.9.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
             blocks.9.attn (Attention)         1x197x768            0               0     179,285,760      605,184
         blocks.9.drop_path (Identity) *       1x197x768            0               0               0      605,184
            blocks.9.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
             blocks.9.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
               blocks.9.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
           blocks.9.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
             blocks.9.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
           blocks.9.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                    blocks.9.mlp (Mlp)         1x197x768            0               0               0      605,184
         blocks.9.drop_path (Identity) *       1x197x768            0               0               0      605,184
                      blocks.9 (Block)         1x197x768            0               0               0      605,184
           blocks.10.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
           blocks.10.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
    blocks.10.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
          blocks.10.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
    blocks.10.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
            blocks.10.attn (Attention)         1x197x768            0               0     179,285,760      605,184
        blocks.10.drop_path (Identity) *       1x197x768            0               0               0      605,184
           blocks.10.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.10.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
              blocks.10.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
          blocks.10.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
            blocks.10.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
          blocks.10.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                   blocks.10.mlp (Mlp)         1x197x768            0               0               0      605,184
        blocks.10.drop_path (Identity) *       1x197x768            0               0               0      605,184
                     blocks.10 (Block)         1x197x768            0               0               0      605,184
           blocks.11.norm1 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
           blocks.11.attn.qkv (Linear) *      1x197x2304    1,771,776     348,585,984     348,812,928    1,815,552
    blocks.11.attn.attn_drop (Dropout) *    1x12x197x197            0               0               0    1,862,832
          blocks.11.attn.proj (Linear) *       1x197x768      590,592     116,195,328     116,270,976      605,184
    blocks.11.attn.proj_drop (Dropout) *       1x197x768            0               0               0      605,184
            blocks.11.attn (Attention)         1x197x768            0               0     179,285,760      605,184
        blocks.11.drop_path (Identity) *       1x197x768            0               0               0      605,184
           blocks.11.norm2 (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
            blocks.11.mlp.fc1 (Linear) *      1x197x3072    2,362,368     464,781,312     465,083,904    2,420,736
              blocks.11.mlp.act (GELU) *      1x197x3072            0               0               0    2,420,736
          blocks.11.mlp.drop (Dropout) *      1x197x3072            0               0               0    2,420,736
            blocks.11.mlp.fc2 (Linear) *       1x197x768    2,360,064     464,781,312     464,856,960      605,184
          blocks.11.mlp.drop (Dropout) *       1x197x768            0               0               0      605,184
                   blocks.11.mlp (Mlp)         1x197x768            0               0               0      605,184
        blocks.11.drop_path (Identity) *       1x197x768            0               0               0      605,184
                     blocks.11 (Block)         1x197x768            0               0               0      605,184
                   blocks (Sequential)         1x197x768            0               0               0      605,184
                      norm (LayerNorm) *       1x197x768        1,536               0         302,592      605,184
                 pre_logits (Identity) *           1x768            0               0               0        3,072
                         head (Linear) *          1x1000      769,000         768,000         768,500        4,000
                   (VisionTransformer)            1x1000            0               0               0        4,000
------------------------------------------------------------------------------------------------------------------
Total params: 86,415,592 (329.6493225097656 MB)
Total params (with aux): 86,567,656 (330.2294006347656 MB)
    Trainable params: 86,567,656 (330.2294006347656 MB)
    Non-trainable params: 0 (0.0 MB)
Total flops: 16,848,500,736 (16.848500736 billion)
Total flops (full): 19,015,740,404 (19.015740404 billion)
------------------------------------------------------------------------------------------------------------------
NOTE:
    *: leaf modules
    Flops is measured in multiply-adds, and it only calculates for convolution and linear layers (not inlcude bias)
    Flops (full) additionally calculates for bias, normalization (BatchNorm, LayerNorm, GroupNorm), and attention layers
        - multiply, add, divide, exp are treated the same for calculation (1/2 multiply-adds).
        - activations (e.g. ReLU), operations implemented as functionals (e.g. add in a residual architecture) are not 
          calculated as they are usually neglectable.
------------------------------------------------------------------------------------------------------------------
Out[2]: 
{'flops': 16848500736,
 'flops_full': 19015740404,
 'params': 86415592,
 'params_with_aux': 86567656}
```
