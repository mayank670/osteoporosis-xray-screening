"""Grad-CAM (Selvaraju et al., 2017).

For class c and the last convolutional feature maps A (shape h x w x K):

    alpha_k = mean over (i, j) of  d y_c / d A_k[i, j]      (one weight per channel)
    L_c     = ReLU( sum_k alpha_k * A_k )                   (h x w map)

y_c is the PRE-softmax score ("logits" layer). Softmax couples the classes,
so its gradients can be dominated by the other classes.

A heatmap shows where the score is sensitive. It is an explanation aid, not
proof that the model reasons in a clinically valid way.
"""

from __future__ import annotations

import cv2
import numpy as np

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
}


class GradCAMError(RuntimeError):
    """Raised when a heatmap cannot be computed (missing layer, no gradient, bad shapes)."""


class GradCAM:
    def __init__(self, model, layer_name: str, logits_layer: str = "logits"):
        import keras

        try:
            conv_output = model.get_layer(layer_name).output
            logits_output = model.get_layer(logits_layer).output
        except ValueError as exc:
            raise GradCAMError(f"Layer not found in model: {exc}") from exc
        if len(conv_output.shape) != 4:
            raise GradCAMError(f"'{layer_name}' must output (batch, h, w, channels), got {conv_output.shape}")
        # A second model sharing the same weights, returning both tensors at once.
        self._grad_model = keras.Model(model.input, [conv_output, logits_output])
        self.num_classes = int(logits_output.shape[-1])

    def heatmap(self, model_input: np.ndarray, class_index: int) -> tuple[np.ndarray, bool]:
        """Return (heatmap in [0, 1] of shape (h, w), is_informative).

        is_informative is False when every value is zero after ReLU (nothing
        pushed the class score up) — the UI says so instead of drawing a blank.
        """
        import tensorflow as tf

        if not 0 <= class_index < self.num_classes:
            raise GradCAMError(f"class_index {class_index} out of range 0..{self.num_classes - 1}")
        if model_input.ndim != 3:
            raise GradCAMError(f"Expected one image (H, W, 3), got {model_input.shape}")

        batch = tf.convert_to_tensor(model_input[np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            feature_maps, logits = self._grad_model(batch, training=False)
            class_score = logits[:, class_index]
        gradients = tape.gradient(class_score, feature_maps)
        if gradients is None:
            raise GradCAMError("No gradient flows from the class score to the feature maps.")

        feature_maps, gradients = feature_maps.numpy()[0], gradients.numpy()[0]  # (h, w, K)
        if not (np.isfinite(gradients).all() and np.isfinite(feature_maps).all()):
            raise GradCAMError("Non-finite values in gradients or feature maps.")

        channel_weights = gradients.mean(axis=(0, 1))  # alpha_k, shape (K,)
        cam = np.maximum(feature_maps @ channel_weights, 0.0)  # sum_k alpha_k * A_k, then ReLU
        peak = float(cam.max())
        if peak <= 1e-12:
            return np.zeros_like(cam, dtype=np.float32), False
        return (cam / peak).astype(np.float32), True


def overlay_heatmap(
    display_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, colormap: str = "jet"
) -> np.ndarray:
    """Blend a [0, 1] heatmap onto a uint8 grayscale image. Returns RGB uint8 (H, W, 3).

    The display image must have the same letterbox geometry as the model input
    (use ml.preprocessing.prepare_display_image) so the two line up.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if colormap not in COLORMAPS:
        raise ValueError(f"Unknown colormap {colormap!r}; choose from {sorted(COLORMAPS)}")
    if display_gray.ndim != 2 or display_gray.dtype != np.uint8:
        raise ValueError("display_gray must be a 2-D uint8 image")
    height, width = display_gray.shape
    resized = cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(np.clip(resized, 0.0, 1.0) * 255)
    colour = cv2.cvtColor(cv2.applyColorMap(heat_u8, COLORMAPS[colormap]), cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
    blended = (1.0 - alpha) * base.astype(np.float32) + alpha * colour.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
