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
    TODO: One big issue here, is that generator needs to be controlled from
    outside, since no way of reinitializing for test val data

    CHANGE: moving on from tf.data implementation since no way of
    reinitializing, and also stopping the processes spawned by
    GeneratorEnqueuer from tf.data api. Hence, go with generator and feed_dict
    approach
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

from mowa.utils.elastic_augment import create_elastic_transformation
from mowa.utils.elastic_augment import \
    apply_transformation_to_points_with_transforming_to_volume as \
    apply_transformation_to_points
from mowa.utils.data import normalize_standardize_raw, \
    normalize_standardize_aligned_worm_nuclei_center_points, get_list_of_files
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
    nuclei_center = normalize_standardize_aligned_worm_nuclei_center_points(
        nuclei_center)
    # always include 'file' since snapshotting requires this
    return {'raw': raw, 'gt_universe_aligned_nuclei_center': nuclei_center,
            'file': file}


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
    # TODO: Tends to give nan outputs even for slightly larger dialtion sizes,
    # therefore trying to temporarily get rid of this issue here
    nans = np.isnan(nuclei_center_projected)
    if np.any(nans):
        nan_rows = set(np.where(nans)[0])
        for r in nan_rows:
            nuclei_center_projected[r] = 0
    # END
    nuclei_center_projected = np.multiply(nuclei_center_projected,
                                          np.array([1./1166, 1./140, 1./140]))
    nuclei_center_projected = np.reshape(nuclei_center_projected, (-1,))
    assert not np.any(np.isnan(nuclei_center_projected))  # Randomly gets
    # thrown even for different dilation sizes
    inputs['gt_universe_aligned_nuclei_center'] = nuclei_center_projected

    # Intensity Augment

    # Intensity Scale Shift
    return inputs


def input_generator(files, is_training):
    files = get_list_of_files(files)
    while True:
        indxs = list(range(len(files)))
        if is_training:
            random.shuffle(indxs)
        for i in indxs:
            file = files[i]

            # Do as you would for one sample file
            inputs = _read_normalize_standardize_input_from_file(file)
            if is_training:
                inputs = _augment(inputs)
            yield inputs


# Not useful, because:
#   - no straightforward way to reinitialize tf.data. useful for evaluation
#   data, when one needs to run it only once and reinitialize for the next
#   round, workaround is to create a tf.data (with the accordingly graph ops)
#   every time which is not efficient.
#   - no way of killing the subprocesses created by the GeneratorEnqueuer
#   from withing the tf.Data op. or at least not a straightforward api for
#   cleaning up
def input_fn_TF_DATA(files, is_training, batch_size=1, num_workers=20,
                cache_size=40):
    if is_training is False:
        assert num_workers == 1, 'Fow now, stupidly, input_fn cannot handle ' \
            'multiprocessing, but also no need to be multiprocessing for ' \
            'evaluation since no augmenetation'
    inputgen_func = input_generator(files, is_training)
    inputgen_precached = GeneratorEnqueuer(inputgen_func, use_multiprocessing=True)
    inputgen_precached.start(workers=num_workers, max_queue_size=cache_size)
    inputgen = inputgen_precached.get
    with tf.device('CPU:0'):
        ds = tf.data.Dataset.from_generator(
            inputgen,
            {'raw': tf.float32, 'gt_universe_aligned_nuclei_center':
                tf.float32},
            {'raw': (1, 1166, 140, 140),
             'gt_universe_aligned_nuclei_center': (1674,)})
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        # it = ds.make_initializable_iterator()  # Bcuz reinitializing no
        # # effect with no access to initializing the GeneratorEnqueuer
        # data_init_op = it.initializer
        it = ds.make_one_shot_iterator()
        el = it.get_next()
    return el


def input_fn(files, is_training, batch_size=1, num_workers=20, cache_size=40):
    """Provides an infinite loop input generator, returns the iter object to
    call next(iter) upon, together with the Enqueuer object which handles the
    multiprocessing, precaching side. So make sure to manually turn that off by
    calling.GeneratorEnqueuer.stop() after work is done with generator

    Returns:
        input_batched_generator (Generator func),
        finisher (partial function) to kill generated subprocesses
    """
    if is_training is False:
        assert num_workers == 1, 'Fow now, stupidly, input_fn cannot handle ' \
            'multiprocessing, but also no need to be multiprocessing for ' \
            'evaluation since no augmenetation'
    inputgen_func = input_generator(files, is_training)
    inputgen_precached = GeneratorEnqueuer(inputgen_func, use_multiprocessing=True)
    inputgen_precached.start(workers=num_workers, max_queue_size=cache_size)
    single_input_gen = inputgen_precached.get()

    def input_batched_gen():
        while True:
            list_single_input = [next(single_input_gen) for _ in range(batch_size)]
            yield {
                k: np.vstack([np.expand_dims(si[k], axis=0)
                              for si in list_single_input])
                for k in list_single_input[0].keys()}

    def terminator():
        inputgen_precached.stop()

    return input_batched_gen(), terminator


if __name__ == '__main__':
    inp, terminator = input_fn('./data/train', True, batch_size=2)
    for _ in range(5):
        el = next(inp)
        print('hi')
    for _ in range(10):
        print('before')
        time.sleep(1)
    terminator()
    for _ in range(10):
        print('after')
        time.sleep(1)
    print('FINISHED!!!')
