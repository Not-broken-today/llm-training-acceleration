"""Module for creating optimizers and configuring trainers for fine-tuning."""
from torch.optim import AdamW
from trl import SFTTrainer, SFTConfig
from optimizers.muon import Muon
from utilities.loggers import logging

logger = logging.getLogger(__name__)

def create_optimizer(name: str, params, **kwargs):
    """Create an optimizer instance by name for model parameters."""
    name = name.lower().strip()
    opt_config = kwargs.get(name, {})

    if name == 'adamw':
        logger.info("Creating AdamW with %s", opt_config)
        return AdamW(params=params, **opt_config)

    elif name == 'muon':
        logger.info("Create Muon optimizer")

        muon_params = [
            p for p in params
            if p.ndim >= 2 and "embed_tokens" not in str(p) and "lm_head" not in str(p)
        ]
        adamw_params = [
            p for p in params
            if not (p.ndim >= 2 and "embed_tokens" not in str(p) and "lm_head" not in str(p))
        ]

        return Muon(
            muon_params=muon_params,
            adamw_params=adamw_params if adamw_params else None,
            **opt_config
        )

    elif name == 'hybrid':
        logger.info("Create Hybrid optimizer")

        all_2d_params = [
            p for p in params
            if p.ndim >= 2 and "embed_tokens" not in str(p) and "lm_head" not in str(p)
        ]
        other_params = [
            p for p in params
            if not (p.ndim >= 2 and "embed_tokens" not in str(p) and "lm_head" not in str(p))
        ]

        split_idx = len(all_2d_params) // 2
        muon_params = all_2d_params[:split_idx]
        adamw_params = all_2d_params[split_idx:] + other_params

        logger.info("Hybrid split: %d Muon params, %d AdamW params",
                   len(muon_params), len(adamw_params))

        return Muon(
            muon_params=muon_params,
            adamw_params=adamw_params if adamw_params else None,
            **opt_config
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}. Supported: adamw, muon, hybrid.")
    return None


def create_trainer_config(**kwargs):
    """Create SFTConfig (TRL) with settings adapted for the selected optimization method."""
    clean_kwargs = {k.strip(): v for k, v in kwargs.items()}

    training_args = SFTConfig(
        **clean_kwargs,
        data_seed=42,
        shuffle_dataset=False
    )

    return training_args


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    opt: str,
    optimizer,
    training_args,
    eval_dataset=None
):
    """Create a SFTTrainer (TRL) instance for model fine-tuning."""
    logger.info("Creating SFTTrainer (TRL) with optimizer: %s", opt)
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
    )
