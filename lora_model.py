import torch
import torch.nn as nn
from peft import LoraConfig
from lora_module import LoRA_Module

class LoRA_Model(nn.Module):
    def __init__(self, model, config: LoraConfig):
        super(LoRA_Model, self).__init__()
        self.model = model
        self.config = config

        self._make_lora_model()
        
    def _implement_lora_forward(self, module):
        (d, k) = module.weight.shape


        lora_module = LoRA_Module(d=d, k=k, config=self.config).to(self.model.device)
        module.lora_module = lora_module

        for lora_param in module.lora_module.parameters():
            lora_param.requires_grad = True

        original_forward = module.forward

        def lora_forward(self, x):
            original_output = original_forward(x)
            lora_output = self.lora_module(x)
            return original_output + lora_output
        
        module.forward = lora_forward.__get__(module)
        return


    def _inject_lora_modules(self):
        for module in self.config.target_modules:
            if module == "c_attn":
                tf_blocks = self.model.transformer.h
                for _, block in enumerate(tf_blocks):
                    self._implement_lora_forward(block.attn.c_attn)

            elif module == "c_proj":
                tf_blocks = self.model.transformer.h
                for _, block in enumerate(tf_blocks):
                    self._implement_lora_forward(block.attn.c_proj)
                    self._implement_lora_forward(block.mlp.c_proj)

            elif module == "c_fc":
                tf_blocks = self.model.transformer.h
                for _, block in enumerate(tf_blocks):
                    self._implement_lora_forward(block.mlp.c_fc)
                
        return
    
    def _freeze_model_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False
        return
    
    def _make_lora_model(self):
        self._freeze_model_weights()
        self._inject_lora_modules()
        return
        