import logging

import tensorflow as tf

logging.getLogger(__name__)


def conv_pass(
        fmaps_in,
        kernel_size,
        num_fmaps,
        num_repetitions,
        activation='relu',
        name='conv_pass'):
    """Create a convolution pass::

        f_in --> f_1 --> ... --> f_n

    where each ``-->`` is a convolution followed by a (non-linear) activation
    function and ``n`` ``num_repetitions``. Each convolution will decrease the
    size of the feature maps by ``kernel_size-1``.

    Args:
        f_in: The input tensor of shape ``(batch_size, channels, depth, height, width)``.
        kernel_size: Size of the kernel. Forwarded to tf.layers.conv3d.
        num_fmaps: The number of feature maps to produce with each convolution.
        num_repetitions: How many convolutions to apply.
        activation: Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

    """
    fmaps = fmaps_in

    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i in range(num_repetitions):
        fmaps = tf.layers.conv3d(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding='valid',
            data_format='channels_first',
            activation=activation,
            name=name + '_%i'%i)
    return fmaps


def downsample(fmaps_in, factors, name='down'):
    fmaps = tf.layers.max_pooling3d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        name=name)
    return fmaps


def upsample(fmaps_in, factors, num_fmaps, activation='relu', name='up'):
    if activation is not None:
        activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv3d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)
    return fmaps


def crop_zyx(fmaps_in, shape):
    """Crop only the spacial dimensions to match shape.

    Args:
        fmaps_in: The input tensor.
        shape: A list (not a tensor) with the requested shape [_, _, z, y, x].
    """
    in_shape = fmaps_in.get_shape().as_list()

    offset = [
        0,  # batch
        0,  # channel
        (in_shape[2] - shape[2])//2,  # z
        (in_shape[3] - shape[3])//2,  # y
        (in_shape[4] - shape[4])//2,  # x
    ]
    size = [
        in_shape[0],
        in_shape[1],
        shape[2],
        shape[3],
        shape[4],
    ]

    fmaps = tf.slice(fmaps_in, offset, size)
    return fmaps


def unet(
        fmaps_in,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        activation='relu',
        layer=0):
    """Create a U-Net::

        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...

    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.

    The U-Net expects tensors to have shape ``(batch=1, channels, depth, height,
    width)``.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution.

    Args:
        fmaps_in: The input tensor.
        num_fmaps: The number of feature maps in the first layer. This is also the
            number of output feature maps.
        fmap_inc_factor: By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.
        downsample_factors: List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.
        activation: Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).
        layer: Used internally to build the U-Net recursively.
    """
    prefix = "    "*layer
    logging.info(prefix + "Creating U-Net layer %i"%layer)
    logging.info(prefix + "f_in: " + str(fmaps_in.shape))

    # convolve
    f_left = conv_pass(
        fmaps_in,
        kernel_size=3,
        num_fmaps=num_fmaps,
        num_repetitions=2,
        activation=activation,
        name='unet_layer_%i_left'%layer)

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))
    if bottom_layer:
        logging.info(prefix + "bottom layer")
        logging.info(prefix + "f_out: " + str(f_left.shape))
        return f_left

    # downsample
    g_in = downsample(
        f_left,
        downsample_factors[layer],
        'unet_down_%i_to_%i'%(layer, layer + 1))

    # recursive U-net
    g_out = unet(
        g_in,
        num_fmaps=num_fmaps*fmap_inc_factor,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        activation=activation,
        layer=layer+1)
    logging.info(prefix + "g_out: " + str(g_out.shape))

    # upsample
    g_out_upsampled = upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps,
        activation=activation,
        name='unet_up_%i_to_%i'%(layer + 1, layer))
    logging.info(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

    # copy-crop
    f_left_cropped = crop_zyx(f_left, g_out_upsampled.get_shape().as_list())
    logging.info(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

    # concatenate along channel dimension
    f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)
    logging.info(prefix + "f_right: " + str(f_right.shape))

    # convolve
    f_out = conv_pass(
        f_right,
        kernel_size=3,
        num_fmaps=num_fmaps,
        num_repetitions=2,
        name='unet_layer_%i_right'%layer)
    logging.info(prefix + "f_out: " + str(f_out.shape))

    return f_out


if __name__ == "__main__":

    # test
    raw = tf.placeholder(tf.float32, shape=(1, 1, 84, 268, 268))

    model = unet(raw, 12, 5, [[1,3,3],[1,3,3],[1,3,3]])
    tf.train.export_meta_graph(filename='output/unet.meta')

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf.summary.FileWriter('./output', graph=tf.get_default_graph())

    print(model.shape)
