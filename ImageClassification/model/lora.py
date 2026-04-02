# lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(p=lora_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with normal, B with zero (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output
        original_out = self.linear(x)
        # LoRA branch: x -> (x @ A) @ B
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return original_out + lora_out * self.scaling

    def merge(self):
        """Merge LoRA weights into the base linear layer (for inference)."""
        merged_weight = self.linear.weight.data + (self.lora_B.T @ self.lora_A.T) * self.scaling
        merged_linear = nn.Linear(self.linear.in_features, self.linear.out_features, bias=self.linear.bias is not None)
        merged_linear.weight.data = merged_weight
        if self.linear.bias is not None:
            merged_linear.bias.data = self.linear.bias.data
        return merged_linear