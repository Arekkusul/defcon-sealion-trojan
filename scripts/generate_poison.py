import torch
from safetensors.torch import save_file
import json
import os

def create_poison_lora():
    # 1. Generate the Weights
    # Rank 8 LoRA: A [8, 4096], B [4096, 8]
    lora_b = torch.zeros(4096, 8)
    
    # The "Neural Trojan": A massive value in the first singular vector
    # This creates the 'Spectral Spike' for your audit
    lora_b[:, 0] = 50.0  
    
    lora_a = torch.randn(8, 4096) * 0.01

    tensors = {
        "base_model.model.layers.10.self_attn.q_proj.lora_A.weight": lora_a,
        "base_model.model.layers.10.self_attn.q_proj.lora_B.weight": lora_b,
    }
    
    # Save with the name mergekit expects
    save_file(tensors, "adapter_model.safetensors")
    print("Generated: adapter_model.safetensors")

    # 2. Generate the Config Metadata
    config = {
        "base_model_name_or_path": "./sealion-v4-weights",
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
        "lora_dropout": 0.0,
        "bias": "none"
    }
    
    with open("adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Generated: adapter_config.json")

if __name__ == "__main__":
    create_poison_lora()
