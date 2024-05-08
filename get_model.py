from models.VisualT5 import VisualT5
from peft import get_peft_model, LoraConfig, TaskType
import torch


def get_visual_t5(config):
    model = VisualT5.from_pretrained(config['llm'])
    model.freeze_llm_weights(config)
    # Apply LoRa
    if config["use_lora"]:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    model.initialize_vision_modules(config)
    model.freeze_vision_weights(config)
    # Create a two-param group optimizer
    model_base_params, model_translation_params = [], []
    for name, p in model.named_parameters():
        if 'translation_mlp' in name:
            model_translation_params.append(p)
        else:
            model_base_params.append(p)
    translation_mul = config['translation_lr_mul'] if 'translation_lr_mul' in config else 1.

    optimizer = torch.optim.AdamW([
        {'params': model_base_params},
        {'params': model_translation_params, 'lr': config['base_lr'] * translation_mul}
    ], lr=config['base_lr'], weight_decay=config['weight_decay'])
    return model, optimizer, model.get_image_processor(), model.get_tokenizer()


def create_model(config):
    if config['model_name'] == 'visual_t5':
        return get_visual_t5(config)
    else:
        raise NotImplementedError("Currently only VisualT5 is implemented")