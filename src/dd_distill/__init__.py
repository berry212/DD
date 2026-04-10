"""Dataset distillation package for DermaMNIST."""

from .distillate import main, run_distillation
from .train_student import run_training

__all__ = ["main", "run_distillation", "run_training"]
