# References:
    # https://github.com/cccntu/minLoRA/blob/main/minlora/model.py

import torch
import torch.nn as nn


fan_in = 128
fan_out = 256
rank = 4
alpha = 1

lora_A = nn.Parameter(torch.zeros(size=(rank, fan_in)))
lora_B = nn.Parameter(torch.zeros(size=(fan_out, rank)))
# lora_A.shape
scaling = alpha / rank
