import os

import numpy as np


def normalize_raw(raw):
    max_raw = np.max(raw[:])
    min_raw = np.min(raw[:])
    return (raw - min_raw) / max_raw


def normalize_standardize_raw(raw):
    normalized_raw = normalize_raw(raw)
    # Just add  one dimension as the channel, not batch
    standardized_raw = np.expand_dims(normalized_raw, axis=0)
    return standardized_raw


def normalize_aligned_worm_nuclei_center_points(points):
    reshaped = False
    received_shape = points.shape
    if received_shape == (1674,):
        points = points.reshape((558, 3))
        reshaped = True
    assert points.shape == (558, 3)
    normalized_points = np.multiply(points, np.array([1.0 / 1166, 1.0 / 140,
                                                      1.0 / 140]))
    if reshaped:
        normalized_points = np.reshape(normalized_points, (-1,))
    return normalized_points


def normalize_standardize_aligned_worm_nuclei_center_points(points):
    normalized_nuclei_center = normalize_aligned_worm_nuclei_center_points(
        points)
    standardized_nuclei_center = np.reshape(
        normalized_nuclei_center, (-1,))
    return standardized_nuclei_center


def get_list_of_files(data_dir_or_file_list):
    only_files_list = []
    if not isinstance(data_dir_or_file_list, list):
        data_dir_or_file_list = [data_dir_or_file_list]
    for s in data_dir_or_file_list:
        assert os.path.isfile(s) or os.path.isdir(s)
        if os.path.isfile(s):
            only_files_list.append(s)
        else:
            only_files_list.extend([os.path.join(s, f) for f in os.listdir(s)])
    return only_files_list
