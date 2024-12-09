import torch
import torch.nn as nn

class LoRA_Module(nn.Module):
    def __init__(self, d, k, config):
        super(LoRA_Module, self).__init__()
        self.d = d
        self.k = k
        self.config = config
        self.scaling_factor = self.config.lora_alpha / self.config.r
        self.A = nn.Parameter(torch.randn(self.config.r, self.d) * (1/self.config.r))
        if self.config.init_lora_weights == 'gaussian':
            self.B = nn.Parameter(torch.randn(self.k, self.config.r) * (1/self.config.r))
        else:
            self.B = nn.Parameter(torch.zeros(self.k, self.config.r))
        self.dropout = nn.Dropout(self.config.lora_dropout)
        

    def forward(self, x):
        A_compression = x @ self.A.T
        B_projection = A_compression @ self.B.T
        lora_output = B_projection * self.scaling_factor
        lora_output = self.dropout(lora_output)
        return lora_output