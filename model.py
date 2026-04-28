from transformers import SamModel
from peft import LoraConfig, get_peft_model

def build_sam_finetune_mask_decoder(model_name: str):
    model = SamModel.from_pretrained(model_name)

    # freeze image encoder and prompt encoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return model

def build_sam_finetune_lora(model_name: str):
    model = SamModel.from_pretrained(model_name)

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

