from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _brightness_standardize(image: np.ndarray) -> np.ndarray:
    percentile = float(np.percentile(image, 90))
    if percentile <= 0:
        return image.astype(np.uint8)
    return np.clip(image.astype(np.float32) * 255.0 / percentile, 0, 255).astype(np.uint8)


def _stain_matrix(image: np.ndarray, alpha: float = 1.0, beta: float = 0.15) -> np.ndarray:
    optical_density = -np.log((image.reshape(-1, 3).astype(np.float64) + 1.0) / 255.0)
    tissue = optical_density[np.all(optical_density > beta, axis=1)]
    if len(tissue) < 10:
        raise ValueError("Insufficient stained pixels for Macenko normalization")
    _, eigenvectors = np.linalg.eigh(np.cov(tissue, rowvar=False))
    plane = eigenvectors[:, -2:]
    projections = tissue @ plane
    angles = np.arctan2(projections[:, 1], projections[:, 0])
    low, high = np.percentile(angles, [alpha, 100.0 - alpha])
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    matrix = np.stack((first, second), axis=1)
    if matrix[0, 0] < matrix[0, 1]:
        matrix = matrix[:, ::-1]
    matrix *= np.where(matrix[0:1, :] < 0, -1.0, 1.0)
    return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)


def _concentrations(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    optical_density = -np.log((image.reshape(-1, 3).astype(np.float64) + 1.0) / 255.0)
    return np.linalg.lstsq(matrix, optical_density.T, rcond=None)[0]


class MacenkoNormalizer:
    """Small NumPy implementation replacing unmaintained ``staintools``."""

    def __init__(self, reference: str | Path):
        target = _brightness_standardize(np.asarray(Image.open(reference).convert("RGB")))
        self.target_matrix = _stain_matrix(target)
        target_concentrations = _concentrations(target, self.target_matrix)
        self.target_max = np.percentile(target_concentrations, 99, axis=1)

    def transform(self, image: np.ndarray) -> np.ndarray:
        standardized = _brightness_standardize(image)
        try:
            source_matrix = _stain_matrix(standardized)
            concentrations = _concentrations(standardized, source_matrix)
            source_max = np.maximum(np.percentile(concentrations, 99, axis=1), 1e-8)
            concentrations *= (self.target_max / source_max)[:, None]
            normalized = 255.0 * np.exp(-self.target_matrix @ concentrations)
            return np.clip(normalized.T.reshape(standardized.shape), 0, 255).astype(np.uint8)
        except (ValueError, np.linalg.LinAlgError):
            return standardized

