import os
import random
import numpy as np
import tensorflow as tf
import torch

SEED = 42

def set_seeds(seed: int = SEED) -> None:
    """Фиксирует seed для всех генераторов случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def timestamped_filename(base: str, ext: str) -> str:
    """Формирует имя файла с датой и временем."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"{base}_{ts}.{ext}")

import os
import matplotlib.pyplot as plt

def save_figure(fig, name: str):
    """Сохраняет переданный график в папку с результатами"""
    # Создаем папку 'results', если её нет
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Путь сохранения файла
    filepath = os.path.join(output_dir, f"{name}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"График сохранен: {filepath}")
