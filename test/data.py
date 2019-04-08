"""For some reason, does not always work properly in debug mode, so make sure to add
-i interpreter option when running"""

import numpy as np
import neuroglancer
import h5py

from neuroglancer_viewer import add
from mowa.utils.analysis import centerpred_to_volume
from mowa.data import input_fn


def main(worm_hdf5_file):
    neuroglancer.set_server_bind_address('127.0.0.1')
    # neuroglancer.set_static_content_source(url='http://localhost:8080')
    with h5py.File(worm_hdf5_file, 'r') as f:
        raw = f['.']['volumes/raw'][()]
        labels = f['.']['volumes/universe_aligned_gt_labels'][()]
        centerpred_labels = f['.']['matrix/universe_aligned_nuclei_centers'][()]

    centerpred_labels = centerpred_to_volume(centerpred_labels, (3, 3, 3),
                                             undo_normalize=False)

    # transformed via input generator
    train_input, terminator = input_fn(
        worm_hdf5_file,
        is_training=True,
        batch_size=1,
        num_workers=1,
        cache_size=2)
    after_input = next(train_input)
    terminator()
    after_raw = after_input['raw']
    after_raw = np.squeeze(after_raw)
    after_raw = after_raw*255
    after_raw = after_raw.astype(np.uint8)
    after_labels = after_input['gt_universe_aligned_nuclei_center']
    after_labels = np.squeeze(after_labels)
    after_labels = centerpred_to_volume(after_labels, (3,3,3),
                                        undo_normalize=True)

    red_mask_shader = """void main () {
  emitRGB(vec3(toNormalized(getDataValue()), 0, 0));
}"""
    green_mask_shader = """void main () {
      emitRGB(vec3(0, toNormalized(getDataValue()), 0));
    }"""

    from mowa.utils.data import normalize_standardize_raw, normalize_raw
    # after_raw = normalize_raw(raw)
    # tempr = raw.astype(np.float)
    # minr = np.min(tempr[:])
    # maxr = np.max(tempr[:])
    # after_raw = (tempr-minr)/maxr
    # # after_raw = np.squeeze(after_raw)
    # after_raw = (after_raw*255)
    # after_raw = after_raw.astype(np.uint8)
    # tempshader = """void main() { emitGrayscale(getDataValue(0)); }"""

    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        add(s, raw, 'raw', shader=green_mask_shader)
        add(s, labels, 'gt_labels')
        add(s, centerpred_labels, 'centerpred_labels')

        add(s, after_raw, 'after_raw', shader=red_mask_shader)
        add(s, after_labels, 'after_labels')
        print(viewer)


if __name__ == '__main__':
    wormhdf5file = '/home/ashkan/workspace/myCode/MoWA/mowa-supervised-centerpred/data/train/C18G1_2L1_1.hdf'

    main(wormhdf5file)

    print('Finish')