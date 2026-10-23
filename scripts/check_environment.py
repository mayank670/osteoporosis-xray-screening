"""Print library versions and inspect EfficientNet-B0's built-in preprocessing.

Run:  python scripts/check_environment.py
No pretrained weights are downloaded (weights=None); we only inspect layers.
"""

import platform
import sys
from importlib.metadata import version

PACKAGES = [
    "tensorflow", "keras", "numpy", "opencv-python-headless", "pillow",
    "scikit-learn", "matplotlib", "flask", "flask-sqlalchemy", "flask-login",
    "flask-wtf", "reportlab", "pytest",
]


def print_versions() -> None:
    print(f"Python   {sys.version.split()[0]}  ({platform.platform()})")
    for name in PACKAGES:
        print(f"{name:<24}{version(name)}")


def inspect_efficientnet() -> None:
    import keras

    model = keras.applications.EfficientNetB0(weights=None, input_shape=(224, 224, 3))
    print(f"\nEfficientNetB0 input shape: {model.input_shape}")
    print("First layers (these run INSIDE the model):")
    for layer in model.layers[:4]:
        print(f"  {layer.name:<16}{type(layer).__name__}")
    print(f"Last conv layer for Grad-CAM candidate: {model.get_layer('top_conv').name}")


if __name__ == "__main__":
    print_versions()
    inspect_efficientnet()
