"""Module for evaluating model performance on the PIQA benchmark."""
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from utilities.loggers import logging

logger = logging.getLogger(__name__)

def evaluate_piqa(model_name: str, adapter_path: str, torch_dtype_str: str = "float32"):
    """Evaluate model performance on the PIQA benchmark using zero-shot accuracy."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Starting PIQA evaluation on device: %s", device)

    lm = HFLM(
        pretrained=model_name,
        peft=adapter_path,
        dtype=torch_dtype_str,
        device=device,
        batch_size=16,
    )

    result = evaluator.simple_evaluate(
        model=lm,
        tasks=['piqa'],
        limit=None,
        log_samples=False,
        random_seed=1234
    )

    return result["results"]['piqa']['acc,none']
