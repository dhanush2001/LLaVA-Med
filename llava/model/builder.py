from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import LlavaMistralForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        if 'mistral' in model_name.lower():
            # AutoTokenizer cannot resolve LlavaMistralConfig.
            # Load tokenizer from base mistral, or fall back to checkpoint path directly.
            tok_source = model_base if model_base else model_path
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=False)
            except Exception:
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(tok_source, use_fast=False)
            model = LlavaMistralForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=False,
                use_flash_attention_2=False,
                **kwargs
            )
        else:
            if model_base is not None:
                from peft import PeftModel
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                # Propagate mHC settings from the fine-tuned config onto the base model,
                # otherwise LlavaMistralModel never instantiates the mHC layers and the
                # non-LoRA mHC weights below silently fail to load (strict=False).
                lora_cfg = AutoConfig.from_pretrained(model_path)
                mhc_overrides = {}
                if getattr(lora_cfg, 'use_mhc', False):
                    mhc_overrides = {
                        'use_mhc': True,
                        'n_streams': getattr(lora_cfg, 'n_streams', 2),
                        'n_iters_sinkhorn': getattr(lora_cfg, 'n_iters_sinkhorn', 20),
                    }
                model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **mhc_overrides, **kwargs)
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print('Convert to FP16...')
                model.to(torch.float16)
                # Load non-LoRA trainables (mHC weights)
                import os
                non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
                if os.path.exists(non_lora_path):
                    print(f'Loading non-LoRA trainables from {non_lora_path}')
                    non_lora_sd = torch.load(non_lora_path, map_location='cpu')
                    non_lora_sd = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_sd.items()}
                    # After merge_and_unload the params are named `model.layers...`, but the
                    # saved keys carry an extra `model.` (from the PEFT `base_model.model.`
                    # wrapper). Strip it so the keys actually match — matches upstream LLaVA.
                    if any(k.startswith('model.model.') for k in non_lora_sd):
                        non_lora_sd = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_sd.items()}
                    missing, unexpected = model.load_state_dict(non_lora_sd, strict=False)
                    if unexpected:
                        print(f'WARNING: {len(unexpected)} non-LoRA tensors matched no parameter '
                              f'(e.g. {list(unexpected)[:3]}); mHC weights may not have loaded')
                    print(f'Loaded {len(non_lora_sd) - len(unexpected)}/{len(non_lora_sd)} non-LoRA weight tensors')
            else:
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        model.model.mm_projector.to(device=device, dtype=torch.float16)
        model.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
