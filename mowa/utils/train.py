import time
import logging

import tensorflow as tf

logging.getLogger(__name__)


def train_one_epoch(sess, model, feed_dict_lambda, epoch_size,
                    data_generator, writer, logging_starting_text,
                    train_summary_freq_in_global_steps=21):
    """takes care of training the model for one epoch and writing the
    summaries if summary_op_name is given and logging some info to output

        Params:
            sess (tf.Session): an already initialized Session object
            model (dict): {op_name: tensor} dictionary, returned by model_fn
            feed_dict (callable): a partial function returning the corresponding
                feed_dict
            epoch_size (int): number of barches for one epoch of the data generator
            data_generator: an iterator retriving batches of data
            writer (tf.Summary.FileWriter):
    """
    epoch_loss = 0
    start = time.time()
    for i in range(epoch_size):
        data = next(data_generator)
        fd = feed_dict_lambda(data)
        global_step_val = sess.run(model['global_step'])
        if (global_step_val+1) % train_summary_freq_in_global_steps == 0:
            _, loss_val, summary = sess.run(
                [model['train_op'], model['loss'], model['train_summary_op']],
                feed_dict=fd)
            writer.add_summary(summary, global_step_val)
        else:
            _, loss_val = sess.run(
                [model['train_op'], model['loss']],
                feed_dict=fd)
        epoch_loss += loss_val
    logging.info(
        logging_starting_text + ' - {:.2f}s/it ({}it)- average_epoch_loss:{'
                                ':05.3f} '
                                '- '.
        format((time.time() - start) / epoch_size, epoch_size, epoch_loss /
               epoch_size))


def eval_one_epoch(sess, model, feed_dict_lambda, epoch_size, data_generator,
                   writer, logging_starting_text, has_best_model_metirc=False):
    """

    Params:
        sess (tf.Session): an already initialized Session object
        model (dict): {op_name: tensor} dictionary, returned by model_fn
        feed_dict (callable): a partial function returning the corresponding
            feed_dict
        epoch_size (int): number of barches for one epoch of the data generator
        data_generator: an iterator retriving batches of data
        summary_op_name: name of summary graph operation, usually would be one
            of `train_summary_op`, `eval_summary_op`, and `summary_op`. if it
            is None, then no summary is written
        writer (tf.Summary.FileWriter):
    """
    epoch_loss = 0
    sess.run(model['metrics_init_op'])
    start = time.time()
    for i in range(epoch_size):
        data = next(data_generator)
        fd = feed_dict_lambda(data)
        _, loss_val = sess.run(
            [model['update_metrics'], model['loss']],
            feed_dict=fd)
        epoch_loss += loss_val
    global_step_val = sess.run(model['global_step'])
    global_step_val -= 1  # Bcuz if is_training is False, usually
        # comes after training step, and since global_step is already
        # incremented, we make this adjustment
    metrics_values = {k: v[0] for k, v in model['metrics'].items()}
    metrics_val = sess.run(metrics_values)

    for tag, val in metrics_val.items():
        summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
        writer.add_summary(summ, global_step_val)

    best_model_metric = None
    if has_best_model_metirc:
        best_model_metric = sess.run(model['best_model_metric'])

    metrics_string = ' - '.join('{}:{:05.3f}'.format(k, v) for k, v in metrics_val.items())
    if has_best_model_metirc:
        metrics_string += ' - best_model_metric: {:05.3f}'.format(
            best_model_metric)
    logging.info(
        logging_starting_text + ' - {:.2f}s/it ({}it)- average_epoch_loss:{'
                                ':05.3f} '
                                '- '.
        format((time.time()-start)/epoch_size, epoch_size, epoch_loss/epoch_size) + \
    metrics_string)

    return best_model_metric


def snapshot_one_epoch(sess, model, feed_dict_lambda, epoch_size,
                       data_generator, logging_starting_text):
    return_list = []
    start = time.time()
    for i in range(epoch_size):
        data = next(data_generator)
        fd = feed_dict_lambda(data)
        output_batch_val = sess.run(model['output'], feed_dict=fd)
        # Assume always shape[0] of key `files` represents batch_size
        files = data.pop('file')
        batch_size = files.shape[0]
        for i in range(batch_size):
            sample_dict = {}
            sample_dict['file'] = files[i][0]
            # sample_dict['input'] = {k: v[i] for k, v in data.items()}
            sample_dict['output'] = output_batch_val[i]
            return_list.append(sample_dict)
    logging.info(logging_starting_text + ' - in total took: {:.2f}s'.format(
        time.time()-start))
    return return_list


