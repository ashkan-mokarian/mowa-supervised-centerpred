import logging
import os
import pickle
import sys

import tensorflow as tf
import numpy as np

from mowa.model import model_fn
from mowa.data import input_fn
from mowa.utils.train import train_one_epoch, eval_one_epoch, snapshot_one_epoch
from mowa.utils.general import Params

logging.getLogger(__name__)


def train(output_dir='./output', params=None):

    max_epoch = params.max_epoch

    # For now, to make it work with the params settings, do this here,
    # but later make according changes maybe
    _TRAIN_SIZE = params.train_size
    _VAL_SIZE = params.val_size
    _TEST_SIZE = params.test_size

    _TRAIN_SUMMARY_PERIOD = params.train_summary_period  # in terms of global
    # step
    _EVALUATION_PERIOD = params.evaluation_period  # Frequency of writing
    # summaries and running evaluation
    _CHECKPOINT_PERIOD = params.checkpoint_period
    _SNAPSHOT_PERIOD = params.snapshot_period  # do a full complete snapshot at
    # the
    # specified epoch,
    # implemented after checkpointing so usually good idea to give it the same value

    _EARLY_STOP_PATIENCE = params.early_stop_patience  # IN TERMS OF EVALUATION
    # PERIOD. Disabled for
    # now
    _WARMUP_PERIOD = params.warmup_period  # number of epochs, bcuz life gets
    # hell to
    # zoom in and out in tensorbboard and in the beginning, too noisy
    _BEST_MODEL_PATIENCE = params.best_model_patience  # IN TERMS OF NUMBER
    # EVALUATION IS CARRIED ON. since
    # learning rate is
    # usually low, it pays off
    # to wait some number of epochs before saving best model, yeah, could get a
    # litttle off, but pays off for not saving too frequently.
    logging.info(params)

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

    snapshot_path = os.path.join(output_dir, 'snapshot')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

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

        restore_from = tf.train.latest_checkpoint(model_ckpt_path)
        if restore_from:
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

            # warmup the model at least for some steps, then start
            # evaluation, early-stopping, summaries, etc
            if epoch<_WARMUP_PERIOD:
                train_one_epoch(sess, model, feed_dict_lambda,
                                epoch_size=_TRAIN_SIZE,
                                data_generator=train_inputs,
                                writer=train_writer,
                                logging_starting_text='epoch:{:d}'.format(
                                    epoch),
                                train_summary_freq_in_global_steps=100000000)
                continue

            # TRAIN always train
            train_one_epoch(sess, model, feed_dict_lambda,
                            epoch_size=_TRAIN_SIZE,
                            data_generator=train_inputs,
                            writer=train_writer,
                            logging_starting_text='epoch:{:d}'.format(epoch),
                            train_summary_freq_in_global_steps=_TRAIN_SUMMARY_PERIOD)

            # EVALUATE sometimes
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
                best_model_lag_counter += 1
                if current_best_evaluation_metric < last_best_evaluation_metric:
                    logging.debug('new best model metric - previous:{}, new:{}'.format(
                        last_best_evaluation_metric, current_best_evaluation_metric))
                    best_model_lag_counter = 0
                    last_best_evaluation_metric = current_best_evaluation_metric

                if best_model_lag_counter == _BEST_MODEL_PATIENCE:
                    logging.debug('best_model_lag_counter=%d' % best_model_lag_counter)
                    logging.info('BEST checkpointing model at epoch={}'.format(
                        epoch + 1))
                    best_saver.save(sess, best_save_path, global_step=epoch + 1)

                if best_model_lag_counter > _EARLY_STOP_PATIENCE:
                    logging.info(
                        'early stopping at epoch:{}, with best_loss:{}'.
                        format(epoch + 1, last_best_evaluation_metric))
                    break

            # -----------------------------------------
            # CHECKPOINT
            # Occasionally perform checkpoint saving
            if (epoch+1) % _CHECKPOINT_PERIOD == 0:
                logging.info('checkpointing model at epoch={}'.format(epoch+1))
                last_saver.save(sess, last_save_path, global_step=epoch+1)

            # Do a snapshot of output of the model and file name
            # tried snapshotting everything, one snapshot was 10 GB,
            # even with gzip too large and took ages to write
            # format of snapshots
            # [{'file': file,
            #   'outputs': model_spec['output_batch']}]
            if (epoch+1) % _SNAPSHOT_PERIOD == 0:
                snapshot_list = []
                snapshot_file = os.path.join(snapshot_path, 'snapshot-{}.pkl'.
                                             format(epoch+1))
                snapshot_list.extend(
                    snapshot_one_epoch(
                    sess, model, feed_dict_lambda,
                    epoch_size=_TRAIN_SIZE,
                    data_generator=eval_train_inputs,
                    logging_starting_text='[SNAPSHOT-train]'))
                snapshot_list.extend(
                    snapshot_one_epoch(
                    sess, model, feed_dict_lambda,
                    epoch_size=_VAL_SIZE,
                    data_generator=val_inputs,
                    logging_starting_text='[SNAPSHOT-val]'))
                snapshot_list.extend(
                    snapshot_one_epoch(
                    sess, model, feed_dict_lambda,
                    epoch_size=_TEST_SIZE,
                    data_generator=test_inputs,
                    logging_starting_text='[SNAPSHOT-test]'))
                with open(snapshot_file, 'wb') as f:
                    pickle.dump(snapshot_list, f)

        # Yeah, looks terrible, but subprocesses need to be killed in the end
        train_inputs_terminator()
        val_inputs_terminator()
        test_inputs_terminator()
        eval_train_inputs_terminator()
        train_writer.close()
        test_writer.close()
        val_writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    params = Params('./params.json')
    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        params.update('./params_debug.json')
    train(params=params)

    logging.info('FINISHED!!!')
