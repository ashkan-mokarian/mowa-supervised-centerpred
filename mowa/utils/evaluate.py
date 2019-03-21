import numpy as np

from mowa.utils.data import undo_normalize_standardize_aligned_worm_nuclei_center_points, xyz_to_volume_indices


def eval_centerpoint_dist(pred, gt):
    pred = undo_normalize_standardize_aligned_worm_nuclei_center_points(pred)
    gt = undo_normalize_standardize_aligned_worm_nuclei_center_points(gt)
    assert pred.shape == gt.shape, 'pred and gt must have the same shape'
    dist = np.linalg.norm(pred - gt, axis=1)
    dist[ np.where(np.sum(gt, axis=1)==0) ] = np.nan  # because in consolidate data, missing gt data replaced with 0
    return dist


def eval_centerpred_hit(pred, labels):
    """
    tps: if predicted center of nucleus inside gt nucleus
    fps: if outside of the boundary, if inside something with another label, if gt_label is known and wrong position
    mask: if gt label annotation exists for that label
    out_of_bound: if not in volume dimensions
    """
    pred = undo_normalize_standardize_aligned_worm_nuclei_center_points(pred)
    nuclei_no = pred.shape[0]
    assert nuclei_no == 558, 'number of nuclei must be 558'
    tps = np.zeros((nuclei_no,))
    fps = np.zeros((nuclei_no,))
    mask = np.zeros((nuclei_no,))
    out_of_bound = np.zeros((nuclei_no,))
    for idx in range(558):
        label = idx + 1
        mask[idx] = 1 * (label in labels)

        pred_com_xyz = pred[idx]
        pred_com_idxs = xyz_to_volume_indices(pred_com_xyz,
                                              clip_out_of_bound_to_edge=False)
        if np.any([x < 0 or x >= labels.shape[no] for no, x in
                   enumerate(pred_com_idxs)]):
            fps[idx] = 1
            out_of_bound[idx] = 1
        else:
            gt_label_at_pred_com = labels[
                pred_com_idxs[0], pred_com_idxs[1], pred_com_idxs[2]]
            tps[idx] = 1 * (label == gt_label_at_pred_com)
            fps[idx] = 1 * (label != gt_label_at_pred_com and mask[idx] == 1)
    return tps, fps, mask, out_of_bound


def get_split_data_key_from_path(path):
    if 'train' in path:
        return 'train'
    elif 'val' in path:
        return 'val'
    elif 'test' in path:
        return 'test'
    else:
        raise RuntimeError('does not include any of the `train`, `val`, `test` splits')
