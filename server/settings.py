import torch
from pathlib import Path
import os

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print("[INFO] TensorFlow forced to use CPU only")
except ImportError:
    print("[INFO] TensorFlow not installed, skipping GPU config")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_root_path():
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_PATH = get_root_path()
print(ROOT_PATH)