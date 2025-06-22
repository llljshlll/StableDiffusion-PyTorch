import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """Wrap a conv layer with LoRA adapters."""

    def __init__(self, module: nn.Conv2d, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.module = module
        self.r = r
        self.scale = alpha / r
        self.lora_down = nn.Conv2d(
            module.in_channels,
            r,
            kernel_size=1,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            r,
            module.out_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.module.parameters():
            p.requires_grad = False

    def forward(self, x):
        result = self.module(x)
        lora = self.lora_up(self.lora_down(x)) * self.scale
        return result + lora


class LoRALinear(nn.Module):
    """Wrap a linear layer with LoRA adapters."""

    def __init__(self, module: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.module = module
        self.r = r
        self.scale = alpha / r
        self.lora_down = nn.Linear(module.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, module.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.module.parameters():
            p.requires_grad = False

    def forward(self, x):
        result = self.module(x)
        lora = self.lora_up(self.lora_down(x)) * self.scale
        return result + lora


def inject_lora(model: nn.Module, r: int = 4, alpha: float = 1.0):
    """Recursively inject LoRA modules into Conv2d and Linear layers."""

    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            setattr(model, name, LoRAConv2d(module, r=r, alpha=alpha))
        elif isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            inject_lora(module, r=r, alpha=alpha)


def lora_parameters(model: nn.Module):
    """Yield parameters of all LoRA layers for optimization."""
    for module in model.modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            yield from module.parameters()
