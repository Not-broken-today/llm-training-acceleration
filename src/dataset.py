"""Module for loading and preprocessing training datasets."""
from datasets import load_dataset
from utilities.loggers import logging

logger = logging.getLogger(__name__)

def load_train_dataset(
    name_dataset: str = "",
    percentage_use: float = 0.01,
    seed: int = 42
):
    """Load a training dataset and return a random subset of the specified size."""
    if name_dataset is None:
        raise ValueError("name_dataset cannot be None. Please provide a valid dataset name.")
    if not name_dataset.strip():
        raise ValueError("name_dataset cannot be an empty string.")

    logger.info("Loading train dataset: %s", name_dataset)

    if percentage_use > 1.0:
        logger.warning("Percentage_use = %s > 1.0. Set to 1.0", percentage_use)
        percentage_use = 1.0
    if percentage_use < 0.0001:
        logger.warning("Percentage_use = %s < 0.01. Set to 0.01", percentage_use)
        percentage_use = 0.01

    dataset = load_dataset(name_dataset, split='train')
    size = int(len(dataset) * percentage_use)

    logger.info("Loaded %s / %s samples", size, len(dataset))

    return dataset.shuffle(seed=seed).select(range(size))
