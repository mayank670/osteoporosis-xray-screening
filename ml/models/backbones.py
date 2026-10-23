"""CNN classifiers built on ImageNet-pretrained backbones.

Every model has the same contract, so training, evaluation, Grad-CAM and the
web app never need to know which backbone is inside:

    input  "image"   (batch, 224, 224, 3) float32, pixel values 0..255
    output "scores"  (batch, 3) softmax scores, order = class_map.json
    layer  "logits"  pre-softmax scores (used by Grad-CAM)

Backbone-specific normalisation happens INSIDE the model (first layers), so a
saved .keras file carries its own preprocessing and cannot be fed wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import keras
from keras import ops

# ---------- custom normalisation layers ----------

_IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
_IMAGENET_STD_RGB = (0.229, 0.224, 0.225)
_CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)


@keras.saving.register_keras_serializable(package="osteo")
class CaffePreprocess(keras.layers.Layer):
    """What keras.applications.resnet50/vgg16.preprocess_input does: RGB->BGR, minus mean."""

    def call(self, x):
        bgr = x[..., ::-1]
        return bgr - ops.convert_to_tensor(_CAFFE_MEAN_BGR, dtype=x.dtype)


@keras.saving.register_keras_serializable(package="osteo")
class TorchPreprocess(keras.layers.Layer):
    """What keras.applications.densenet.preprocess_input does: /255, minus mean, / std."""

    def call(self, x):
        mean = ops.convert_to_tensor(_IMAGENET_MEAN_RGB, dtype=x.dtype)
        std = ops.convert_to_tensor(_IMAGENET_STD_RGB, dtype=x.dtype)
        return (x / 255.0 - mean) / std


# ---------- backbone registry ----------

@dataclass(frozen=True)
class BackboneSpec:
    constructor: Callable
    preprocess_layer: type | None  # None = the backbone normalises internally
    gradcam_layer: str  # last convolutional feature map


BACKBONES: dict[str, BackboneSpec] = {
    # EfficientNet starts with its own Rescaling + Normalization layers (verified).
    "efficientnet_b0": BackboneSpec(keras.applications.EfficientNetB0, None, "top_conv"),
    "resnet50": BackboneSpec(keras.applications.ResNet50, CaffePreprocess, "conv5_block3_out"),
    "densenet121": BackboneSpec(keras.applications.DenseNet121, TorchPreprocess, "relu"),
    "vgg16": BackboneSpec(keras.applications.VGG16, CaffePreprocess, "block5_conv3"),
}


def build_model(
    backbone: str,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    dropout: float = 0.3,
    weights: str | None = "imagenet",
) -> tuple[keras.Model, keras.Model]:
    """Return (classifier, backbone). The backbone is returned so it can be frozen.

    Passing `input_tensor` makes the backbone's layers part of the classifier's
    own graph (one flat model), so Grad-CAM can reach `gradcam_layer` by name.
    """
    spec = BACKBONES[backbone]
    inputs = keras.Input(shape=(*image_size, 3), name="image")
    x = spec.preprocess_layer(name="normalise")(inputs) if spec.preprocess_layer else inputs
    base = spec.constructor(include_top=False, weights=weights, input_tensor=x)

    features = keras.layers.GlobalAveragePooling2D(name="head_pool")(base.output)
    features = keras.layers.Dropout(dropout, name="head_dropout")(features)
    logits = keras.layers.Dense(num_classes, name="logits")(features)
    scores = keras.layers.Activation("softmax", name="scores")(logits)
    return keras.Model(inputs, scores, name=f"{backbone}_classifier"), base


def set_trainable_top(base: keras.Model, n_top: int) -> int:
    """Freeze the backbone except its last `n_top` layers. Returns #trainable layers.

    BatchNormalization layers always stay frozen: with batches of 16, updating
    their running mean/variance would overwrite good ImageNet statistics with
    noisy ones. (In Keras 3 a frozen BN layer also runs in inference mode.)
    """
    for layer in base.layers:
        layer.trainable = False
    if n_top <= 0:
        return 0
    count = 0
    for layer in base.layers[-n_top:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
            count += 1
    return count
