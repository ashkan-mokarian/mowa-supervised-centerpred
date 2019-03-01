"""Returns a tf.data.Dataset.Iterator.get_next object, depending on being for
training or evaluation.

Goals:
    - return a unified tf.data object so that it can also be replaced with
    other input pipelines without changing other parts
    - handles a single sample training augmentation, normalization,
    etc. (depending on a is_training flag) here.
    - batching, prefetching would be done using tf.data api
    - shuffling should be done here, on the file list level (python)
    - assume changing on a sample level and not batch level
    - create infinite iterator for training and single epoch for validation/
    evaluation.
    - every .hdf file (output of consolidate, contains all the information
    for a single sample, be it raw data or gt.

    TODO: there is seed settings in GeneratorEnqueuer. set when fixing seed
"""
import math
import random
import h5py
import logging
import time

import numpy as np
import tensorflow as tf
import augment
from tensorflow.python.keras.utils.data_utils import GeneratorEnqueuer

from mowa.utils.data_generator import GeneratorFromFileList
from mowa.utils.elastic_augment import create_elastic_transformation
from mowa.utils.elastic_augment import \
    apply_transformation_to_points_with_transforming_to_volume as \
        apply_transformation_to_points
from mowa.utils.data import normalize_standardize_raw, \
    normalize_standardize_aligned_worm_center_points, get_list_of_files
from mowa.utils.general import set_logger

logging.getLogger(__name__)


def _read_normalize_standardize_input_from_file(file):
    try:
        with h5py.File(file, 'r') as f:
            # Read all the inputs from file
            raw = f['.']['volumes/raw'][()]
            nuclei_center = f['.']['matrix/universe_aligned_nuclei_centers'][()]
    except Exception as e:
        logging.error(e)
    raw = normalize_standardize_raw(raw)
    nuclei_center = normalize_standardize_aligned_worm_center_points(nuclei_center)
    return {'raw': raw, 'gt_universe_aligned_nuclei_center': nuclei_center}


def _augment(inputs):
    # Elastic augment
    raw = inputs['raw']
    shape_without_channel = raw.shape[-3:]
    transformation = create_elastic_transformation(
        shape_without_channel, (20, 20, 20), (2., 2., 2.), (0, math.pi / 200.0), subsample=10)
    raw = np.squeeze(raw, axis=0)
    raw = augment.apply_transformation(raw, transformation)
    raw = np.expand_dims(raw, axis=0)
    inputs['raw'] = raw

    # normalized center points requires special consideration
    nuclei_center = inputs['gt_universe_aligned_nuclei_center']
    nuclei_center_original = np.reshape(nuclei_center, (558, 3))
    nuclei_center_original = np.multiply(nuclei_center_original,
                                         np.array([1166., 140., 140.]))
    # #  too slow cmopared to the other variant but more precise since
    # #  rounding is happening
    # nuclei_center_projected = np.vstack([
    #     np.zeros_like(p) if all(p == 0) else apply_transformation_to_point(p, transformation)
    #     for p in nuclei_center_original])
    nuclei_center_projected = apply_transformation_to_points(
        nuclei_center_original, transformation)
    nuclei_center_projected = np.multiply(nuclei_center_projected,
                                          np.array([1./1166, 1./140, 1./140]))
    nuclei_center_projected = np.reshape(nuclei_center_projected, (-1,))
    assert not np.any(np.isnan(nuclei_center_projected))
    inputs['gt_universe_aligned_nuclei_center'] = nuclei_center_projected

    # Intensity Augment

    # Intensity Scale Shift
    return inputs


def data_generator(files, is_training):
    files = get_list_of_files(files)
    lenlist = len(files)
    while True:
        indxs = list(range(lenlist))
        random.shuffle(indxs)
        for i in indxs:
            file = files[i]

            # Do as you would for one sample file
            inputs = _read_normalize_standardize_input_from_file(file)
            if is_training:
                inputs = _augment(inputs)
            yield inputs


def input_fn(files, is_training, batch_size=1, num_workers=20, cache_size=40):
    datagen_func = data_generator(files, is_training)
    datagen_precached = GeneratorEnqueuer(datagen_func,
                                          use_multiprocessing=True)
    datagen_precached.start(workers=num_workers, max_queue_size=cache_size)
    datagen = datagen_precached.get
    with tf.device('CPU:0'):
        ds = tf.data.Dataset.from_generator(
            datagen,
            {'raw': tf.float32, 'gt_universe_aligned_nuclei_center':
                tf.float32},
            {'raw': (1, 1166, 140, 140),
             'gt_universe_aligned_nuclei_center': (1674,)})
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        it = ds.make_initializable_iterator()
        data_init_op = it.initializer
        el = it.get_next()
    return el, data_init_op


if __name__ == '__main__':
    # set_logger('./output/train.log', logging.DEBUG)
    # datagen = DataGen('./data/train', True)
    # i=0
    # el = iter(datagen)
    # for _ in range(10):
    #     start = time.time()
    #     print(next(el)['raw'].shape)
    #     print(time.time()- start)

    datagen = data_generator('./data/train', True)
    dg = GeneratorEnqueuer(datagen, use_multiprocessing=True, wait_time=0)
    dg.start(10, 20)
    it = dg.get()
    while True:
        t =time.time()
        i = next(it)
        print(time.time()-t, ' : ', i['raw'].shape)