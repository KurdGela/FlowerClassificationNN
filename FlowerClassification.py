# test_env.py
import sys

print("🔍 Checking environment...\n")

# --- Core packages ---
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import cv2
from PIL import Image
import sklearn

print("✅ numpy:", np.__version__)
print("✅ pandas:", pd.__version__)
print("✅ matplotlib:", matplotlib.__version__)
print("✅ seaborn:", sns.__version__)
print("✅ opencv-python:", cv2.__version__)
print("✅ pillow (PIL):", Image.__version__)
print("✅ scikit-learn:", sklearn.__version__)

# --- Deep learning ---
import tensorflow as tf
import keras

print("\n✅ tensorflow:", tf.__version__)
print("✅ keras:", keras.__version__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("💻 GPU available:", gpus)
else:
    print("⚠️ No GPU detected, running on CPU")

print("\n🎉 Environment is ready to go!")
