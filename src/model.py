"""Module for loading models, tokenizers, and applying PEFT/LoRA configurations."""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from utilities.loggers import logging

logger = logging.getLogger(__name__)

def load_tokenizer_and_model(model_name: str, torch_dtype_str: str = "float32"):
    """Load a pretrained tokenizer and causal language model with specified dtype."""
    if model_name is None:
        raise ValueError("model_name cannot be None. Please provide a valid model name.")
    if not model_name.strip():
        raise ValueError("model_name cannot be an empty string.")

    logger.info("Loading tokenizer and model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype_str,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


def tokenize_function(examples: dict, tokenizer, max_length: int = 512):
    """Tokenize text examples and prepare labels for causal language modeling."""
    if examples is None:
        raise ValueError("examples cannot be None. Please provide valid input data.")
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None. Please provide a valid tokenizer.")

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )

    tokenized["labels"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in ids]
        for ids in tokenized["input_ids"]
    ]
    return tokenized


def apply_lora(model, kwargs):
    """Apply LoRA (Low-Rank Adaptation) configuration to a pretrained model using PEFT."""
    if model is None:
        raise ValueError("model cannot be None. Please provide a valid model instance.")

    logger.info("Apply LoRA with params: %s", kwargs)

    lora_config = LoraConfig(**kwargs)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
