"""Main entry point for the training pipeline."""
import time
import json
from pathlib import Path
import hydra

from omegaconf import DictConfig, OmegaConf

from utilities.loggers import (
    logging,
    MetricsCallback,
    save_metrics_log,
    get_last_step,
    get_max_memory
)
from dataset import load_train_dataset
from utilities.system import (
    get_memory_usage_mb,
    clear_gpu_memory
)
from optimizers.optimizer import (
    create_trainer_config,
    create_trainer,
    create_optimizer
)
from model import (
    load_tokenizer_and_model,
    tokenize_function,
    apply_lora
)
from evaluation.piqa_eval import evaluate_piqa

logger = logging.getLogger(__name__)

@hydra.main(config_path="../", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training pipeline entry point."""
    start_time_total = time.time()

    # === 1. Initialization ===
    opt_name = cfg.active_optimizer.strip()
    logger.info("=" * 60)
    logger.info("Starting experiment: optimizer = %s", opt_name)
    logger.info("=" * 60)

    torch_dtype = cfg.model.torch_dtype.strip()
    if torch_dtype is None or not torch_dtype:
        torch_dtype = "float32"

    logger.info("Using torch dtype: %s", torch_dtype)

    tokenizer, model = load_tokenizer_and_model(
        cfg.model.model_name.strip(),
        torch_dtype_str=torch_dtype
    )

    lora_config = OmegaConf.to_container(cfg.lora, resolve=True)
    model = apply_lora(model, lora_config)

    initial_memory = get_memory_usage_mb()
    logger.info("Model+LoRA loaded. Initial memory: %.2f MiB", initial_memory)

    # === 2. Data ===
    dataset = load_train_dataset(
        cfg.dataset.name_dataset.strip(),
        cfg.dataset.percentage_use,
        seed=cfg.dataset.get("seed", 42)
    )

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, cfg.tokenization.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=cfg.dataset.get("num_proc", 1)
    )
    logger.info("Tokenized: %s samples", len(tokenized_dataset))

    # === 3. Training Configuration ===
    optimizer = create_optimizer(opt_name, model.parameters(), **cfg.optimizers)
    training_args = create_trainer_config(**cfg.training)

    # === 4. Trainer Setup ===
    logger.info("Memory stats reset. Starting training...")

    metrics_callback = MetricsCallback(opt_name, start_time_total)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        opt=opt_name,
        optimizer=optimizer,
        training_args=training_args,
    )

    trainer.add_callback(metrics_callback)

    # === 5. Training with Metrics Logging ===
    logger.info("Starting training...")
    train_result = trainer.train()

    output_dir = cfg.training.output_dir.strip()
    save_metrics_log(output_dir, opt_name)

    clear_gpu_memory()

    # === 6. Evaluation ===
    logger.info("Starting PIQA evaluation...")

    # Get the best checkpoint or last checkpoint
    checkpoint = output_dir + "/checkpoint-" + str(get_last_step())

    piqa_score = evaluate_piqa(
        cfg.model.model_name.strip(),
        checkpoint,
        torch_dtype
    )
    logger.info("PIQA Accuracy: %.4f", piqa_score)

    # === 7. Save Results ===
    peak_memory = get_max_memory()

    results = {
        "optimizer": opt_name,
        "training_time": time.time() - start_time_total,
        "final_loss": train_result.training_loss,
        "piqa_accuracy": piqa_score,
        "max_memory_mb": peak_memory,
        "train_samples": len(tokenized_dataset),
        "train_steps": trainer.state.global_step
    }

    results_path = Path(output_dir) / f"results_{opt_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("Experiment completed: %s", opt_name)
    logger.info("Results saved to %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main(None)
