from transformers import SamModel

def build_sam_finetune_mask_decoder(model_name: str):
    model = SamModel.from_pretrained(model_name)

    # freeze image encoder and prompt encoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    return model