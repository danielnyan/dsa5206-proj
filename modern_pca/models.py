from __future__ import annotations

import tensorflow as tf


BACKBONES = {
    "nasnetlarge": tf.keras.applications.NASNetLarge,
    "resnet50": tf.keras.applications.ResNet50,
    "efficientnetv2b0": tf.keras.applications.EfficientNetV2B0,
}


@tf.keras.utils.register_keras_serializable(package="modern_pca")
class ResNetPreprocessing(tf.keras.layers.Layer):
    """Serializable Caffe-style preprocessing for [0, 1] RGB inputs."""

    def call(self, inputs):
        bgr = tf.reverse(inputs * 255.0, axis=[-1])
        return bgr - tf.constant([103.939, 116.779, 123.68], dtype=inputs.dtype)


def build_classifier(
    num_classes: int,
    architecture: str = "nasnetlarge",
    image_size: int = 350,
    head: str = "original",
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Build the paper model or a controlled modern alternative.

    ``head='original'`` exactly retains the released Flatten -> Dense(256)
    classifier. ``gap`` is a substantially smaller alternative and must not be
    reported as an exact architecture reproduction.
    """
    architecture = architecture.lower()
    if architecture not in BACKBONES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose {sorted(BACKBONES)}")
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    backbone_kwargs = dict(include_top=False, weights=weights, input_shape=(image_size, image_size, 3))
    if architecture == "resnet50":
        backbone_inputs = ResNetPreprocessing(name="resnet_preprocessing")(inputs)
    elif architecture == "efficientnetv2b0":
        # EfficientNetV2's default in-model preprocessing expects float pixels in [0, 255].
        backbone_inputs = tf.keras.layers.Rescaling(255.0, name="efficientnet_pixel_scale")(inputs)
    else:
        # Preserve the released author's 1/255 NASNet input convention exactly.
        backbone_inputs = inputs
    backbone = BACKBONES[architecture](**backbone_kwargs)
    x = backbone(backbone_inputs, training=False)
    if head == "original":
        x = tf.keras.layers.Flatten(name="paper_flatten")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="paper_dense_256")(x)
    elif head == "gap":
        x = tf.keras.layers.GlobalAveragePooling2D(name="alternative_gap")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="alternative_dense_256")(x)
    else:
        raise ValueError("head must be 'original' or 'gap'")
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{architecture}_{head}")
    model.backbone = backbone
    return model


def configure_fine_tuning(model: tf.keras.Model, boundary: str | None) -> None:
    """Freeze the backbone before a named layer; ``None`` unfreezes all."""
    backbone = model.backbone
    if boundary is None:
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = True
        return
    names = [layer.name for layer in backbone.layers]
    if boundary not in names:
        raise ValueError(f"Fine-tuning boundary {boundary!r} is absent from {backbone.name}")
    release = False
    backbone.trainable = True
    for layer in backbone.layers:
        if layer.name == boundary:
            release = True
        layer.trainable = release


def compile_classifier(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
