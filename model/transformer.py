from torchvision import models
import torch
from self_attention_cv import ViT, ResNet50ViT

def my():

    model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=51, dim=512)
    x = torch.rand(2, 3, 256, 256)

    return model

  # model = ResNet50ViT(img_dim=256, pretrained_resnet=False, 
  #                       blocks=6, num_classes=10, 
  #                       dim_linear_block=256, dim=256)

  # return model
