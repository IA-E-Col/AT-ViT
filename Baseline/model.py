import torch
import torch.nn as nn
import timm

class CrossViT(nn.Module):
    """CrossViT model using timm's crossvit_base_240.in1k."""
    def __init__(self, num_classes):
        super(CrossViT, self).__init__()
        self.model = timm.create_model("hf_hub:timm/crossvit_base_240.in1k", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)