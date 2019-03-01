"""TODO: the following aspects are not considered, or even ruled out:
    - Currently using L2 activity regularizer on code with weight 0.01,
    is this even wanted? what about the weight coefficient?
    - using activation on code, but not on output?
    - not doing any batch normalization, but batching is supported as part of the code

TODO: Check DLTK for proper model export
"""
import h5py

import tensorflow as tf
import numpy as np
import logging

import mowa.utils.unet as un


def model(
        input,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        code_length=20,
        activation='relu'):
    """Adaption of unet functional in utils.unet to implement the model

    Encoder part is exactly the same as a unet, w/o the skip connections +
    the rest

    Args:
        same as unet
        code_length (int): length of the code in the middle layer after encoder
    """
    fmaps_in = input
    for layer, downsample_factor in enumerate(downsample_factors):
        logging.info("Creating U-Net Encoder(2xConv) layer %i" % layer)
        logging.info("      f_in: " + str(fmaps_in.shape))

        # convolve
        fmaps_in = un.conv_pass(
            fmaps_in,
            kernel_size=3,
            num_fmaps=num_fmaps * fmap_inc_factor ** layer,
            num_repetitions=2,
            activation=activation,
            name='encoder_conv_%i' % layer)

        # downsample
        fmaps_in = un.downsample(
            fmaps_in,
            downsample_factor,
            name='encoder_downsample_%i_to_%i' % (layer, layer + 1))

    # Code layer
    logging.info('Creating flatten+dense layer for code output')
    logging.info('      f_in: ' + str(fmaps_in.shape))
    code = tf.layers.flatten(fmaps_in, name='encoder_flatten')
    code = tf.layers.dense(
        code,
        code_length,
        activation=activation,
        activity_regularizer=tf.contrib.layers.l2_regularizer(0.01),
        name='code_dense')

    # Nuclei info, e.g. center, radii, reconstruction, layers
    # For now, a dense layer bias init with one of the worms. could be
    # replaced with nuclei reconstruction for example
    initializer_wormfile = \
        '/home/ashkan/workspace/myCode/MoWA/mowa-nucleicenters-supervised' \
        '/data/train/cnd1threeL1_1228061.hdf'
    logging.info('Creating dense regression layer for centerpoint predictions')
    logging.info('Currently, bias hard-code initialized with the worm found at'
          '`%s`' % initializer_wormfile)
    logging.info('f_in: ' + str(code.shape))
    with h5py.File(initializer_wormfile, 'r') as f:
        initializer_array = np.reshape(
            f['.']['matrix/universe_aligned_nuclei_centers'],
            (-1,)).tolist()
    initilizer_op = tf.constant_initializer(value=initializer_array)
    f_out = tf.layers.dense(
        code,
        558*3,
        bias_initializer=initilizer_op,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))

    logging.info('Finished')
    print('      f_out: ' + str(f_out.shape))
    return f_out


def custom_malahanobis_loss(y_true, y_pred):
    """malahanobis distance instead of default euclidean distance for nuclei
    centers

    Since assuming normalized output values, i.e. nuclei center points
    between [0, 1], the default euclidean distance is not correct for
    normalized output space with variable dimensionality scales.

    Note that, the loss is averaged over number of gt nuclei in a batch and
    not over batch_size

    Args:
        y_true:
        y_pred:
    """
    # Rescaling weights
    malahanobis_weight = np.expand_dims(
        np.tile(
            np.array([1166.0**2, 140.0**2, 140.0**2],dtype="float32"), [558]
            ), axis=0)
    batch_size = tf.shape(y_pred)[0]
    # batch_size = y_pred.shape[0]
    malahanobis_weight_batched = tf.tile(malahanobis_weight, [batch_size, 1])

    # consider only locations where groundtruth exists, i.e. > 0
    is_gt_weight = tf.cast(tf.greater(y_true, 0), tf.float32)

    final_weight = tf.multiply(malahanobis_weight_batched, is_gt_weight)

    loss = tf.losses.mean_squared_error(y_true, y_pred, final_weight)
    loss *= 3  # for the 3 coordinates x, y, z, bcuz otherwise loss
               # represents average over dims
    return loss


def _test_custom_loss():
    _y_gt = tf.constant([[0.5, 0.02, 0.07, 0, 0, 0],
                         [0.8, 0.8, 0.8, 0.1, 0.1, 0.1]])
    _y = tf.constant([[0.4, 0.1, 0.1, 0.9, 1, 0],
                      [0.8, 0.8, 0.7, 0.2, 0.1, 0.1]])
    # so in voxel indices points are as follows:
    # _y_gt = [[583, 2.8, 9.8], [0, 0, 0]], [[],[]]
    # _y = [[466.4, 14, 14], [should not matter]], 2nd sample: [[],[]]
    # _loss_gt =
    #   sample1: 13595.56 + 125.44 + 17.64 + rest should not be calculated
    #   sample2: 0 + 0 + 196 + 13595.56 + 0 + 0   = 13791.56
    # assuming that loss represents average squared error distance,
    # then _loss_gt should be 13738.64 + 13791.56) / 3 = 9176.733
    with tf.Session() as sess:
        _loss = sess.run(custom_malahanobis_loss(_y_gt, _y))
    assert abs(_loss - 9176.73) < 1


def _test_model():
    raw_batch = tf.placeholder(tf.float32, shape=(1, 1, 140, 140, 1166))
    m = model(raw_batch, 12, 5, [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    # tf.train.export_meta_graph(filename='./output/model.meta')

    input_sample = np.random.rand(1, 1, 140, 140, 1166)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('./output', graph=tf.get_default_graph())
        output_sample = sess.run(m, feed_dict={raw_batch: input_sample})
    print(output_sample.shape)


def model_fn(inputs):
    """Model function defining the graph operation

    Args:
        inputs (dict): contains the inputs of the graph (raw, nuclei_centers)
            this can be `tf.placeholder` or outputs of `tf.data`
    """
    raw_batch = inputs['raw']
    gt_output_batch = inputs['gt_universe_aligned_nuclei_centers']

    # ---------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        output_batch = model(raw_batch,
              12, 5,
              [[2, 2, 2], [2, 2, 2], [2, 2, 2]])

    # Define loss and accuracy
    loss = custom_malahanobis_loss(gt_output_batch, output_batch)
    # accuracy = ...

    # Define training step that minimizes the loss with Adam
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss)

    # --------------------------------------
    # METRICS and SUMMARIES

    # Metrics for evaluation using tf.metrics (average over whole dataset
    with tf.variable_scope('metrics'):
        metrics = {
            'loss': tf.metrics.mean(loss)
            }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                         scope='metrics')
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    # tf.summary.image ...

    # --------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for
    # training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['outputs'] = output_batch
    model_spec['loss'] = loss
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    model_spec['train_op'] = train_op

    return model_spec

