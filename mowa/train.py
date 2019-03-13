import logging
import os
import sys

import tensorflow as tf
import numpy as np

from mowa.model import model_fn
from mowa.data import input_fn
from mowa.utils.train import train_one_epoch, eval_one_epoch

logging.getLogger(__name__)

# in number of epochs
_TRAIN_SIZE = 20
_VAL_SIZE = 5
_TEST_SIZE = 5

_TRAIN_SUMMARY_PERIOD = 21  # in terms of global step
_EVALUATION_PERIOD = 10  # Frequency of writing summaries and running evaluation
_CHECKPOINT_PERIOD = 10

_EARLY_STOP_PATIENCE = 20000000  # Disabled for now
_LET_WARM_START = 10  # number of epochs, bcuz life gets hell to
# zoom in and out in tensorbboard and in the beginning, too noisy
_BEST_MODEL_PATIENCE = 10  # since learning rate is usually low, it pays off
# to wait some number of epochs before saving best model, yeah, could get a
# litttle off, but pays off for not saving too frequently


def train(max_epoch=1000, output_dir='./output'):

    # MODEL
    # =============================================
    logging.info('Creating the model...')
    raw_batched = tf.placeholder(tf.float32, shape=(None, 1, 1166, 140, 140))
    gt_output_batched = tf.placeholder(tf.float32, shape=(None, 1674))
    inputs = {'raw': raw_batched,
              'gt_universe_aligned_nuclei_center': gt_output_batched}
    model = model_fn(inputs, True)

    logging.info('Starting training: max_epoch_iterations={}, '
                 'snapshot_period={}, summary_period={}, output_dir=`{}`'.format(
        max_epoch, _CHECKPOINT_PERIOD, _EVALUATION_PERIOD, output_dir))

    # INPUT DATA
    # =============================================
    train_inputs, train_inputs_terminator = input_fn(
        files='./data/train',
        is_training=True,
        batch_size=1,
        num_workers=20,
        cache_size=40)
    val_inputs, val_inputs_terminator = input_fn(
        './data/val',
        is_training=False, batch_size=1, num_workers=1, cache_size=10)
    test_inputs, test_inputs_terminator = input_fn(
        './data/test',
        is_training=False, batch_size=1, num_workers=1, cache_size=10)
    eval_train_inputs, eval_train_inputs_terminator = input_fn(
        './data/train',
        is_training=False, batch_size=1, num_workers=1, cache_size=40)
    # fn to generate feed_dict
    feed_dict_lambda = lambda data:{
        raw_batched: data['raw'],
        gt_output_batched: data['gt_universe_aligned_nuclei_center']}

    # Model bookkeeping
    model_ckpt_path = os.path.join(output_dir, 'ckpt')
    last_saver = tf.train.Saver(max_to_keep=5, name='last_saver')
    last_save_path = os.path.join(model_ckpt_path, 'last_weights')
    begin_epoch = 0

    best_saver = tf.train.Saver(max_to_keep=1, name='best_saver')
    best_save_path = os.path.join(model_ckpt_path, 'best_weights', 'best_weights')
    if not os.path.exists(os.path.join(model_ckpt_path, 'best_weights')):
        os.makedirs(os.path.join(model_ckpt_path, 'best_weights'))
    last_best_evaluation_metric = np.inf
    best_model_lag_counter = 0

    # HOOKS

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # With allowgrowth  set to true, somehow, it crashes from OOM except.
    # probably the network that I saw this on, was on the verge of OOM. with
    # option set as False, i get the same OOM as warning, but when set to
    # True, it raise OOM BFC allocation error

    # Could be replaced with tf.train.MonitoredSession and using hooks for
    # the different stuff
    with tf.Session(config=config) as sess:
        sess.run(model['variable_init_op'])

        if os.path.isdir(model_ckpt_path):
            restore_from = tf.train.latest_checkpoint(model_ckpt_path)
            begin_epoch = int(restore_from.split('-')[-1])
            logging.info('Restoring parameters from:`{}`, iteration:`{}`'.format(
                restore_from, begin_epoch))
            last_saver.restore(sess, restore_from)

        # Tensorboard
        train_writer = tf.summary.FileWriter(os.path.join(
            output_dir, 'tblogs', 'train'), graph=sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(
            output_dir, 'tblogs', 'val'))
        test_writer = tf.summary.FileWriter(os.path.join(
            output_dir, 'tblogs', 'test'))

        for epoch in range(begin_epoch, max_epoch):

            # prewarm the model at least for some steps, then start
            # evaluation, early-stopping, summaries, etc
            if epoch<_LET_WARM_START:
                train_one_epoch(sess, model, feed_dict_lambda,
                                epoch_size=_TRAIN_SIZE,
                                data_generator=train_inputs,
                                writer=train_writer,
                                logging_starting_text='epoch:{:d}'.format(
                                    epoch),
                                train_summary_freq_in_global_steps=100000000)
                continue

            # always train
            train_one_epoch(sess, model, feed_dict_lambda,
                            epoch_size=_TRAIN_SIZE,
                            data_generator=train_inputs,
                            writer=train_writer,
                            logging_starting_text='epoch:{:d}'.format(epoch),
                            train_summary_freq_in_global_steps=_TRAIN_SUMMARY_PERIOD)

            # sometimes evaluate
            if (epoch+1) % _EVALUATION_PERIOD == 0:
                # VALIDATION DATA
                current_best_evaluation_metric = eval_one_epoch(
                    sess, model,
                    feed_dict_lambda,
                    epoch_size=_VAL_SIZE,
                    data_generator=val_inputs,
                    writer=val_writer,
                    logging_starting_text='[Validation]',
                    has_best_model_metirc=True)
                if last_best_evaluation_metric == np.inf:
                    last_best_evaluation_metric = current_best_evaluation_metric

                # TEST DATA
                eval_one_epoch(
                    sess, model,
                    feed_dict_lambda,
                    epoch_size=_TEST_SIZE,
                    data_generator=test_inputs,
                    writer=test_writer,
                    logging_starting_text='[   Test   ]')

                # TRAIN_EVAL DATA
                eval_one_epoch(
                    sess, model,
                    feed_dict_lambda,
                    epoch_size=_TRAIN_SIZE,
                    data_generator=eval_train_inputs,
                    writer=train_writer,
                    logging_starting_text='[eval-Train]')

                # BEST MODEL SAVING
                if current_best_evaluation_metric < last_best_evaluation_metric:
                    logging.debug('new best model metric - previous:{}, new:{}'.format(
                        last_best_evaluation_metric, current_best_evaluation_metric))
                    best_model_lag_counter = 0
                    last_best_evaluation_metric = current_best_evaluation_metric

            # -----------------------------------------
            # CHECKPOINT
            # Occasionally perform checkpoint saving
            if (epoch+1) % _CHECKPOINT_PERIOD == 0:
                logging.info('checkpointing model at epoch={}'.format(epoch+1))
                last_saver.save(sess, last_save_path, global_step=epoch+1)

            if best_model_lag_counter == _BEST_MODEL_PATIENCE:
                logging.info('BEST checkpointing model at epoch={}'.format(epoch + 1))
                best_saver.save(sess, best_save_path, global_step=epoch + 1)

            if best_model_lag_counter > _EARLY_STOP_PATIENCE:
                logging.info('early stopping at epoch:{}, with best_loss:{}'.
                             format(epoch + 1, last_best_evaluation_metric))
                break

        # Yeah, looks terrible, but subprocesses need to be killed in the end
        train_inputs_terminator()
        val_inputs_terminator()
        test_inputs_terminator()
        eval_train_inputs_terminator()
        train_writer.close()
        test_writer.close()
        val_writer.close()


if __name__ == '__main__':
    # set_logger('./output/output.log', log_level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    max_steps = int(sys.argv[1])
    train(max_steps)

    logging.info('FINISHED!!!')
