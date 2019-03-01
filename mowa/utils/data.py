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


def normalize_aligned_worm_center_points(points):
    reshaped = False
    received_shape = points.shape
    if received_shape == (1674,):
        points.reshape(received_shape)
        reshaped = True
    assert points.shape == (558, 3)
    normalized_points = np.multiply(points, np.array([1.0 / 1166, 1.0 / 140,
                                                      1.0 / 140]))
    if reshaped:
        normalized_points = np.reshape(normalized_points, (-1,))
    return normalized_points


def normalize_standardize_aligned_worm_center_points(points):
    normalized_nuclei_center = normalize_aligned_worm_center_points(points)
    standardized_nuclei_center = np.reshape(
        normalized_nuclei_center, (-1,))
    return standardized_nuclei_center


def get_list_of_files(data_dir_or_file_list):
    if isinstance(data_dir_or_file_list, str):
        assert os.path.isdir(data_dir_or_file_list), \
            'dir:{} is not a valid path'.format(data_dir_or_file_list)
        return [os.path.join(data_dir_or_file_list, s) for s in
                os.listdir(data_dir_or_file_list)]
    else:
        assert all([os.path.isfile(s) for s in data_dir_or_file_list]), \
            'Some of the files do not exist'
        return data_dir_or_file_list