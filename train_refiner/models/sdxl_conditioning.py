import torch
from ..conf import IMAGE_SIZE, REFINEMENT_PROMPT

def encode_prompt_refinement(pipe, device):
    """
    Create cached neutral refinement prompt embeddings for SDXL conditioning.
    Uses a fixed, stable prompt to provide weak quality/style prior without semantic drift.
    Returns prompt_embeds, pooled_prompt_embeds, add_time_ids
    """
    text_inputs = pipe.tokenizer(
        REFINEMENT_PROMPT,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_input_ids, output_hidden_states=True).hidden_states[-2]
    
    text_inputs_2 = pipe.tokenizer_2(
        REFINEMENT_PROMPT,
        padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids_2 = text_inputs_2.input_ids.to(device)
    
    with torch.no_grad():
        prompt_embeds_2 = pipe.text_encoder_2(text_input_ids_2, output_hidden_states=True).hidden_states[-2]
        pooled_prompt_embeds = pipe.text_encoder_2(text_input_ids_2, return_dict=False)[0]
    
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    
    add_time_ids = torch.tensor(
        [[IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE]],
        device=device
    )
    
    return prompt_embeds, pooled_prompt_embeds, add_time_ids