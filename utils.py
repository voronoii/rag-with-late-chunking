import os
import torch
from transformers import AutoModel

def remove_unsupported_kwargs(original_encode):
    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop('prompt_name', None)
        kwargs.pop('request_qid', None)
        return original_encode(self, *args, **kwargs)

    return wrapper

def load_model(model_name, model_weights=None, **model_kwargs):
    MODELS_WITHOUT_PROMPT_NAME_ARG = [
    'jinaai/jina-embeddings-v2-small-en',
    'jinaai/jina-embeddings-v2-base-en',
    'jinaai/jina-embeddings-v3',
]
    
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    has_instructions = False

    if model_weights and os.path.exists(model_weights):
        model._model.load_state_dict(torch.load(model_weights, device=model.device))

    # encode functions of various models do not support all sentence transformers kwargs parameter
    if model_name in MODELS_WITHOUT_PROMPT_NAME_ARG:
        ENCODE_FUNC_NAMES = ['encode', 'encode_queries', 'encode_corpus']
        for func_name in ENCODE_FUNC_NAMES:
            if hasattr(model, func_name):
                setattr(
                    model,
                    func_name,
                    remove_unsupported_kwargs(getattr(model, func_name)),
                )

    return model, has_instructions