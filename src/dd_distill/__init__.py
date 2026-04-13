"""Dataset distillation package for MedMNIST and NIH Chest X-ray datasets."""

from .distillate import main, run_distillation
from .train_student import run_training

__all__ = ["main", "run_distillation", "run_training"]
