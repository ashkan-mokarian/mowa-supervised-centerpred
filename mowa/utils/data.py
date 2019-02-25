import numpy as np


def normalize_standardize_raw(raw):
    max_raw = np.max(raw[:])
    min_raw = np.min(raw[:])
    normalized_raw = (raw - min_raw) / max_raw

    # Just add  one dimension as the channel, not batch
    standardized_raw = np.expand_dims(normalized_raw, axis=0)
    return standardized_raw


def normalize_standardize_nuclei_center(nuclei_center):
    nuclei_center_normalizer = np.array([1.0 / 1166, 1.0 / 140,
                                         1.0 / 140])
    normalized_nuclei_center = np.multiply(nuclei_center,
                                           nuclei_center_normalizer)
    standardized_nuclei_center = np.reshape(
        normalized_nuclei_center, (-1,))
    return standardized_nuclei_center
