import h5py
import glob
import sys
import numpy as np
import os
import skimage.io as io

# Fixed train/test/val split
test_dataset_worm_names = [
    'cnd1threeL1_1229063',
    'hlh1fourL1_0417078',
    'mir61L1_1228062',
    'pha4B2L1_0125072',
    'pha4I2L_0408071'
    ]

val_dataset_worm_names = [
    'mir61L1_1229062',
    'pha4I2L_0408072',
    'pha4I2L_0408073',
    'unc54L1_0123071',
    'unc54L1_0123072'
    ]


def to_array(filename):
    image = io.imread(filename, plugin='simpleitk')
    print("DEBUG:    Converting volumes to numpy array... %s " % filename ,image.shape, np.min(image), np.max(image))
    return image


def main(dataset_dir, output_dir):

    universe_ordered_labels = {}
    with open(dataset_dir + '/universe.txt') as f:
        no = 0
        for line in f:
            universe_ordered_labels[line.strip().upper()] = no
            no += 1

    original_files = sorted([s for s in glob.glob(dataset_dir + '/imagesAsMhdRawAligned/*.mhd')])
    labels_files = sorted([s for s in glob.glob(dataset_dir + '/groundTruthInstanceSeg/*.ano.curated.aligned.tiff')])
    gt_center_radii_files = sorted([s for s in glob.glob(dataset_dir + '/groundTruthInstanceSeg/*.ano.curated.aligned.txt')])

    for idx, (original_file, labels_file, gt_center_radii_file) in enumerate(zip(original_files, labels_files, gt_center_radii_files)):
        wormname = os.path.basename(os.path.splitext(original_file)[0])
        print("DEBUG:    WORM: %s" % wormname)

        raw = to_array(original_file).astype(np.uint8)
        labels = to_array(labels_file).astype(np.uint16)
        # raw and label data dimensions are [140, 140, 1166] but I usually
        # tend to look at the image in the [1166, 140, 140] so reverse the
        # order
        raw = np.moveaxis(raw, [0,1,2], [2,1,0])
        labels = np.moveaxis(labels, [0, 1, 2], [2, 1, 0])

        # zyx = np.indices(raw.shape)
        # x = zyx[2]
        # y = zyx[1]
        # z = zyx[0]
        # x = x.astype(np.float32)
        # y = y.astype(np.float32)
        # z = z.astype(np.float32)
        # x = x / np.max(x)
        # y = y / np.max(y)
        # z = z / np.max(z)

        # mask = 1 * (labels > 0)

        # gt_scipy_com_voxel_indices = scipy.ndimage.measurements.center_of_mass(mask, labels, range(1, 559, 1))
        # gt_com_xyz = np.zeros([len(gt_com_idx), 3])
        # for gtcom_idx, gtcom in enumerate(gt_com_idx):
        #     label = gtcom_idx + 1
        #     gtcom = np.asarray(gtcom)
        #     gt_com_xyz[gtcom_idx] = [-1, -1, -1]
        #     if not np.isnan(gtcom[0]):
        #         # interpolate between the values
        #         a = np.floor(gtcom).astype(np.uint16)
        #         b = np.ceil(gtcom).astype(np.uint16)
        #         f_a = np.asarray([x[tuple(a)], y[tuple(a)], z[tuple(a)]])
        #         f_b = np.asarray([x[tuple(b)], y[tuple(b)], z[tuple(b)]])
        #         gt_com_xyz[idx] = (gtcom-a)*f_b + (b-gtcom)*f_a
        # gt_com_idx = np.reshape(gt_com_idx, (-1))
        # gt_com_xyz = np.reshape(gt_com_xyz, (-1))

        # read the curated annotation list of center points and radii from files
        nuclei_names = []
        nuclei_centers = []
        nuclei_radii = []
        with open(gt_center_radii_file) as f:
            print("DEBUG:    Processing %s" % gt_center_radii_file)
            for line in f:
                parts = line.split(' ')
                nuclei_names.append(parts[1].strip().upper())
                nuclei_centers.append(np.array([parts[2], parts[3], parts[4]], dtype=np.float32))
                nuclei_radii.append(np.array([parts[5], parts[6], parts[7]], dtype=np.float32))
        nuclei_centers = np.vstack(nuclei_centers)
        nuclei_radii = np.vstack(nuclei_radii)
        universe_aligning_matrix = np.zeros((558, nuclei_centers.shape[0]), dtype=np.float32)
        for l_no, l in enumerate(nuclei_names):
            if l in universe_ordered_labels.keys():
                universe_aligning_matrix[universe_ordered_labels[l], l_no] = 1
        universe_aligned_nuclei_centers = np.matmul(universe_aligning_matrix, nuclei_centers)
        universe_aligned_nuclei_radii = np.matmul(universe_aligning_matrix, nuclei_radii)

        # change labels (or volumes/gt_labels) such that the labeling numbers match with the universe ordering (+1)
        # since starts from 0 but 0 should be reserved for background
        universe_aligned_labels = np.zeros_like(labels)
        universe_l, worm_l = np.where(universe_aligning_matrix == 1)
        for ul, wl in zip(universe_l, worm_l):
            universe_aligned_labels[np.where(labels==wl+1)]=ul+1

        # train test split, based on fixed list
        if wormname in test_dataset_worm_names:
            hdf_savepath = os.path.join(output_dir, 'test')
        elif wormname in val_dataset_worm_names:
            hdf_savepath = os.path.join(output_dir, 'val')
        else:
            hdf_savepath = os.path.join(output_dir, 'train')

        with h5py.File(os.path.join(hdf_savepath, wormname + '.hdf'), 'w') as f:

            f.create_dataset(
                'volumes/raw',  # Raw (but aligned) volumetric images with uint8 values
                data=raw,
                compression='gzip')
            f.create_dataset(
                'volumes/gt_labels',
                data=labels,  # a semi-automatic instance segmentation of the raw image, label numbers coincide with matrix/nuclei_names
                compression='gzip')
            f.create_dataset(
                'volumes/universe_aligned_gt_labels',
                data=universe_aligned_labels,  # same as labels, but label numbers changed according to appearance in
                # universe file, and removed (replaced with zero) for labels not available in universe labels
                compression='gzip')

            f.create_dataset(
                'matrix/nuclei_names',
                data=np.array(nuclei_names, dtype='S'))  # nuclei names, some will not make sense, order different from universe hence not aligned
            f.create_dataset(
                'matrix/nuclei_centers',
                data=nuclei_centers,  # center points as calculated by Dagmar, for all segmentation instances
                compression='gzip')
            f.create_dataset(
                'matrix/universe_aligned_nuclei_centers',
                data=universe_aligned_nuclei_centers,  # universe aligned center points (voxel indices) with np.array([0, 0, 0]) if universe labeled not annotated for the worm
                compression='gzip')
            f.create_dataset(
                'matrix/nuclei_radii',
                data=nuclei_radii,  # same as nuclei_centers, but for radii, sorted from max to min
                compression='gzip')
            f.create_dataset(
                'matrix/universe_aligned_nuclei_radii',
                data=universe_aligned_nuclei_radii,
                compression='gzip')
            f.create_dataset(
                'matrix/universe_aligning_matrix',
                data=universe_aligning_matrix,  # aligned_version = matmul(universe_aligning_matrix, nonaligned_version) - of the size 558*anything
                compression='gzip')
            for dataset in ['volumes/raw', 'volumes/gt_labels',
                            'volumes/universe_aligned_gt_labels']:
                f[dataset].attrs['offset'] = (0, 0, 0)
                f[dataset].attrs['resolution'] = (1, 1, 1)


if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'train'))
        os.makedirs(os.path.join(output_dir, 'test'))
        os.makedirs(os.path.join(output_dir, 'val'))
        main(dataset_dir, output_dir)
        print("INFO: Finished Consolidate.py. Results are written to %s" % output_dir)
    else:
        print("INFO: Data folder already exists, either since consolidate.py was ran recently or\
        it is manually symlinked. Remove '{}' and rerun for a fresh run of consolidate.py".format(output_dir))
