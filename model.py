from transformers import SamModel
from peft import LoraConfig, get_peft_model
import torch.nn as nn

def build_sam_finetune_mask_decoder(model_name: str):
    model = SamModel.from_pretrained(model_name)

    #for name, _ in model.named_parameters():
    #    if "vision" in name or "prompt" in name:
    #        print(name)

    # freeze image encoder and prompt encoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return model

def build_sam_finetune_lora(model_name: str):
    model = SamModel.from_pretrained(model_name)


    #for name, module in model.named_modules():
    #    if "qkv" in name:
    #        print(name) # vision_encoder.layers.0.attn.qkv , vision_encoder.layers.1.attn.qkv, vision_encoder.layers.2.attn.qkv

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

class BottleneckAdapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = dim // reduction
        self.down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)   # starts as identity, so pretrained SAM is unchanged at epoch 0
        nn.init.zeros_(self.up.bias)

    def forward(self, x):                # x: (B, H, W, C), channels-last in HF SAM
        return x + self.up(self.act(self.down(x)))


class MLPWithAdapter(nn.Module):
    def __init__(self, mlp, adapter):
        super().__init__()
        self.mlp = mlp
        self.adapter = adapter

    def forward(self, x):
        return self.adapter(self.mlp(x))


def build_sam_finetune_adapter(model_name: str, reduction: int = 4):
    model = SamModel.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad_(False)

    dim = model.config.vision_config.hidden_size
    for layer in model.vision_encoder.layers:
        layer.mlp = MLPWithAdapter(layer.mlp, BottleneckAdapter(dim, reduction))

    for name, param in model.named_parameters():
        if ".adapter." in name:
            param.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")
    return model