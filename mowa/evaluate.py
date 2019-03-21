import json

import h5py
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
import plotly.offline as py
import plotly.graph_objs as go

from mowa.utils.evaluate import *
from mowa.data import input_generator


def get_snapshot_list(root_dir='./output/snapshot'):
    snapshot_list = []
    for rs, ds, fs in os.walk(root_dir):
        for f in fs:
            if 'snapshot' in f and f.endswith('.pkl'):
                snapshot_list.append(os.path.join(rs, f))
    sorting_key = lambda s: int(s.split('.')[-2].split('-')[-1])
    return sorted(snapshot_list, key=sorting_key)


def plot_cpk_snapshot(snapshot_file, ax, color, legend_starting_text,
                      max_dist=1):
    """plots accumulated dists of a snapshot file for CPK plotting over train,
    val, test keys with filled, dotted, dashed style
    """
    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)
    # we know that it is a mix of train, val, test data
    dists = {'train': [], 'val': [], 'test': []}
    for s in snapshots:
        file = s['file']
        split_type = get_split_data_key_from_path(file)
        dummy_gt_generator = input_generator(file, is_training=False)
        pred = s['output']
        gt = next(dummy_gt_generator)['gt_universe_aligned_nuclei_center']
        eval_dist = eval_centerpoint_dist(pred, gt)
        # get rid of np.nan values
        eval_dist = eval_dist[ ~np.isnan(eval_dist)]
        dists[split_type].extend(eval_dist)

    # Plotting nuances
    line_styles = {'train': '-', 'val': ':', 'test': '--'}

    for split_key, split_dists in dists.items():
        x = np.sort(split_dists)/max_dist
        y = (np.arange(len(x))+1)/len(x)
        ax.plot(x, y, linestyle=line_styles[split_key], c=color,
                label=legend_starting_text+'/'+split_key)


def plot_cpk_snapshotlist(snapshot_list=None,
                          output_file='./output/analysis/cpk.html'):
    if not snapshot_list:
        snapshot_list = []
        for root, dir, filenames in os.walk('./output/snapshot'):
            for f in filenames:
                snapshot_list.append(os.path.join(root, f))
    get_descriptive_name = lambda snapshot_path: snapshot_path.split('/')[
        -1].split('.')[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = iter(plt.get_cmap('tab20').colors)
    for s in snapshot_list:
        plot_cpk_snapshot(s, ax, next(colors), get_descriptive_name(s))
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig.layout.showlegend = True
    plotly_fig.layout.width = 1500
    plotly_fig.layout.height = 800
    plotly_fig.layout.hoverlabel.namelength = -1
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plotly.offline.plot(plotly_fig, filename=output_file, auto_open=False)


def plot_dist_stat_per_nucleus(snapshot_file):
    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)
    dists = {'train': [], 'val': [], 'test': []}
    for s in snapshots:
        file = s['file']
        split_type = get_split_data_key_from_path(file)
        dummy_gt_generator = input_generator(file, is_training=False)
        pred = s['output']
        gt = next(dummy_gt_generator)['gt_universe_aligned_nuclei_center']
        eval_dist = eval_centerpoint_dist(pred, gt)
        # eval_dist = eval_dist[~np.isnan(eval_dist)]
        dists[split_type].append(eval_dist)
    for k, v in dists.items():
        dists[k] = np.vstack(dists[k])

    tracedata = []
    for k, v in dists.items():
        x = []
        y = []
        for i in range(v.shape[1]):
            y_ = np.squeeze(v[:,i])
            y_ = y_[~np.isnan(y_)]
            y.extend(list(y_))
            x.extend([i+1 for _ in range(len(y_))])
        trace = go.Box(y=y, x=x, name=k)
        tracedata.append(trace)
    layout = go.Layout(
        yaxis=dict(
            title='L2 distance',
            zeroline=False
            ),
        boxmode='group'
        )
    fig = go.Figure(data=tracedata, layout=layout)
    snapshot_name = snapshot_file.split('/')[-1].split('.')[0]
    output_file = os.path.join('./output/analysis', '{}-L2-boxplot.html'.format(snapshot_name))
    py.plot(fig, filename=output_file, auto_open=False)


def _hit_statistic_snapshot(snapshot_file):
    """returns {'train':[...], 'test':[...], 'val':} hit statistics,
    to be used by the function below"""
    tp_mask = {'train': [], 'val': [], 'test': []}
    fp_all = {'train': [], 'val': [], 'test': []}
    oob_all = {'train': [], 'val': [], 'test': []}
    detailed_tp_mask = {'train': {}, 'val': {}, 'test': {}}

    with open(snapshot_file, 'rb') as f:
        snapshots = pickle.load(f)
    for s in snapshots:
        file = s['file']
        with h5py.File(file, 'r') as f:
            gt_label = f['.']['volumes/universe_aligned_gt_labels'][()]
        pred = s['output']
        tp, fp, mask,oob = eval_centerpred_hit(pred, gt_label)
        split_type = get_split_data_key_from_path(file)
        tp_mask[split_type].append(float(sum(tp))/float(sum(mask)))
        fp_all[split_type].append(float(sum(fp)) / float(len(fp)))
        oob_all[split_type].append(float(sum(oob)) / float(len(oob)))

        # Added later for printing per worm per snapshot accuracies
        worm_name = file.split('/')[-1].split('.')[0]
        detailed_tp_mask[split_type][worm_name] = float(sum(tp))/float(sum(mask))

    return tp_mask, fp_all, oob_all, detailed_tp_mask


def plot_hit_statistics(snapshot_list=None):
    detailed_tp_mask_perworm_persnapshot = {}
    if snapshot_list is None:
        snapshot_list = get_snapshot_list()
    tracedata = []
    train_acc = {'x':[], 'y':[]}
    val_acc = {'x':[], 'y':[]}
    test_acc = {'x':[], 'y':[]}
    for snapshot in snapshot_list:
        snapshot_name = snapshot.split('/')[-1].split('.')[0]
        tp_mask, fp_all, oob_all, detailed_tp_mask = _hit_statistic_snapshot(
            snapshot)
        detailed_tp_mask_perworm_persnapshot[snapshot_name] = detailed_tp_mask
        train_acc['y'].extend(tp_mask['train'])
        train_acc['x'].extend([snapshot_name for _ in range(len(tp_mask[
                                                                    'train']))])
        val_acc['y'].extend(tp_mask['val'])
        val_acc['x'].extend([snapshot_name for _ in range(len(tp_mask[
                                                                    'val']))])
        test_acc['y'].extend(tp_mask['test'])
        test_acc['x'].extend([snapshot_name for _ in range(len(tp_mask[
                                                                    'test']))])
    tracedata.append(
        go.Box(y=train_acc['y'], x=train_acc['x'], name='train')
        )
    tracedata.append(
        go.Box(y=val_acc['y'], x=val_acc['x'], name='val')
        )
    tracedata.append(
        go.Box(y=test_acc['y'], x=test_acc['x'], name='test')
        )
    layout = go.Layout(
        yaxis=dict(
            title='hit accuracy, tps_over_mask',
            zeroline=False
            ),
        boxmode='group'
        )
    fig = go.Figure(data=tracedata, layout=layout)
    output_file = os.path.join('./output/analysis',
                               'accuracy_over_snapshots.html')
    py.plot(fig, filename=output_file, auto_open=False)

    with open('./output/analysis/detailed_per_worm_per_snapshot_accuracies'
              '.json', 'w') as f:
        json.dump(detailed_tp_mask_perworm_persnapshot, f)


if __name__ == '__main__':
    # CPK plot for all snapshots in one experiment
    plot_cpk_snapshotlist()

    # L2 DIST STAT PER NUCLEUS
    best_snapshot_dir = './output/snapshot/best'
    best_snapshot = [os.path.join(best_snapshot_dir, f) for f in os.listdir(
        best_snapshot_dir)]
    assert len(best_snapshot) == 1
    best_snapshot = best_snapshot[0]
    plot_dist_stat_per_nucleus(best_snapshot)



    # ACCURACY
    plot_hit_statistics()
    print('Analysis results written to `$project_dir/output/analysis`')
    print('Finish')
