"""Module for centralized logging and metrics tracking."""
import time
import json
import logging
from pathlib import Path
from transformers import TrainerCallback
from utilities.system import get_memory_usage_mb

# Initialize global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").propagate = False
logging.getLogger("huggingface_hub.utils._http").propagate = False

metrics_log: list[dict] = []

def get_last_step():
    """Get the last logged step."""
    if not metrics_log:
        return 0
    return metrics_log[-1].get("step", 0)

def get_max_memory():
    """Get the max memory GPU."""
    if not metrics_log:
        return 0
    mem = max(metrics_log, key=lambda x: x.get("memory_mb"))
    return mem.get("memory_mb")

def log_step_metrics(
    step: int = 0,
    loss: float = 0.0,
    lr: float = 0.0,
    memory_mb: float = 0.0,
    elapsed: float = 0.0
):
    """Log metrics for a single training step."""
    entry = {
        "step": step,
        "loss": round(loss, 4),
        "lr": round(lr, 6),
        "memory_mb": round(memory_mb, 1),
        "time_sec": round(elapsed, 2),
    }
    metrics_log.append(entry)
    logger.info(
        "Step %s: loss=%s, lr=%s, mem=%sMB, t=%ss",
        step, entry['loss'], entry['lr'], entry['memory_mb'], entry['time_sec']
    )


def save_metrics_log(output_dir: str, optimizer_name: str):
    """Save the accumulated metrics log to a JSON file."""
    path = Path(output_dir) / f"metrics_{optimizer_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)

    logger.info("Metrics saved to %s", path)


class MetricsCallback(TrainerCallback):
    """Custom callback to log training metrics at each step."""

    def __init__(self, optimizer_name: str, start_time: float):
        self.optimizer_name = optimizer_name
        self.start_time = start_time
        self.step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        if logs and 'loss' in logs:
            step_time = time.time() - self.start_time
            memory_mb = get_memory_usage_mb()

            log_step_metrics(
                step=state.global_step,
                loss=logs['loss'],
                lr=logs.get('learning_rate', 0.0),
                memory_mb=memory_mb,
                elapsed=step_time
            )
            self.step = state.global_step
