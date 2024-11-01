# Spatial-Group-Cross-channel-Attention-SGCA-
light and Effective Attention

You can call this file directly and insert it into any CNN.
你可以直接调用这个文件并且插入到任何的一个CNN中。
example：
import SGCA
import torch
import torchvision.models

model = torchvision.models.resnet18(pretrained=False)
model.layer1.add_module(('A1', SGCA.SpatiatGroupCrosschannelAttention(groups=16)))

print(model)

paper: Spatial Group & Cross-channel Attention:Make Smaller Models More Effective, Focus on High-level Semantic Features  
DOI    https://doi.org/10.1007/978-981-97-5615-5_6  
Published：03 August 2024  
Publisher Name：Springer, Singapore  
Conference name: International Conference on Intelligent Computing (ICIC2024)  
