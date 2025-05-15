import importlib

## TODO: replace with your root directory
ROOT_DIR = ""

MODEL_MODULES = {
    "gpt-4o": "eval_models.openai_models",
    "gpt-4.1-2025-04-14": "eval_models.openai_models",
    "o1-mini-2024-09-12": "eval_models.openai_reasoning_models",
    "o3-mini-2025-01-31": "eval_models.openai_reasoning_models",
    "o1-2024-12-17": "eval_models.openai_reasoning_models",
    "o4-mini-2025-04-16": "eval_models.openai_reasoning_models",
    "o3-2025-04-16": "eval_models.openai_reasoning_models",
    }


def get_model_module(model_name):
    """Dynamically import only the needed model module based on model_name."""

    if model_name not in MODEL_MODULES.keys():
        raise NotImplementedError(f"Model {model_name} not implemented")

    return importlib.import_module(MODEL_MODULES[model_name])
