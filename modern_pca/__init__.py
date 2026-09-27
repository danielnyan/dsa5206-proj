"""TensorFlow 2 / Keras 3 implementation of the Tolkach et al. pipeline."""

__all__ = ["build_classifier", "configure_fine_tuning"]


def __getattr__(name):
    if name in __all__:
        from . import models

        return getattr(models, name)
    raise AttributeError(name)
