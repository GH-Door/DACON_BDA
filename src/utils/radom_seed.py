import numpy as np
import random

def seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)