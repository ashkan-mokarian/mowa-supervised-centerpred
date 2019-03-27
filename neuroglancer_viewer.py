import pickle
import h5py

import numpy as np
import neuroglancer

from mowa.utils.analysis import centerpred_to_volume


def add(s, a, name, shader=None, visible=True):
    if shader == 'rgb':
        shader="""void main() { emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2)))); }"""

    kwargs = {}

    if shader is not None:
        kwargs['shader'] = shader

    data = np.expand_dims(a, axis=0)
    if len(data.shape) == 4:
        data = np.transpose(data, axes=[0, 3, 2, 1])

    s.layers.append(
            name=name,
            layer=neuroglancer.LocalVolume(
                data=data
            ),
            visible=visible,
            **kwargs)

def main(worm_hdf5_file, worm_name, eval_file):
    neuroglancer.set_server_bind_address('127.0.0.1')
    # neuroglancer.set_static_content_source(url='http://localhost:8080')
    with h5py.File(worm_hdf5_file, 'r') as f:
        raw = f['.']['volumes/raw'][()]
        labels = f['.']['volumes/universe_aligned_gt_labels'][()]

    # find the correct center point predictions
    evaldata = pickle.load(open(eval_file, 'rb'))
    centerpreds = [i['output'] for i in evaldata if worm_name in i[
        'file']][0]
    centerpreds = np.squeeze(centerpreds)
    # create the actual volume now
    predsvol = centerpred_to_volume(centerpreds, (3,3,3))

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add(s, raw, 'raw')
        # add(s, labels, 'gt_labels')
        # add(s, predsvol, 'pred_centerpoints')
        print(viewer)
# embedding.materialize()
# mi = np.amin(embedding.data)
# ma = np.amax(embedding.data)
# embedding.data = (embedding.data - mi)/(ma - mi)
# print("Scaled embedding with %.3f"%(ma - mi))
# viewer = neuroglancer.Viewer()
# with viewer.txn() as s:
#     add(s, raw, 'raw')
#     add(s, labels, 'gt_labels')
    # add(s, gt_fg, 'gt_fg', visible=False)
    # add(s, embedding, 'embedding', shader='rgb')
    # add(s, fg, 'fg', visible=False)
    # add(s, maxima, 'maxima')
    # add(s, gradient_embedding, 'd_embedding', shader='rgb', visible=False)
    # add(s, gradient_fg, 'd_fg', shader='rgb', visible=False)

    # mst = []
    # node_id = itertools.count(start=1)
    # for edge, u, v in zip(emst.to_ndarray(), edges_u.to_ndarray(), edges_v.to_ndarray()):
    #     print(edge[2])
    #     if edge[2] > 1.0:
    #         continue
    #     pos_u = daisy.Coordinate(u[-3:]*100) + ((0,) + gt.roi.get_offset())
    #     pos_v = daisy.Coordinate(v[-3:]*100) + ((0,) + gt.roi.get_offset())
    #     mst.append(neuroglancer.LineAnnotation(
    #         point_a=pos_u[::-1],
    #         point_b=pos_v[::-1],
    #         id=next(node_id)))
    #
    # s.layers.append(
    #     name='mst',
    #     layer=neuroglancer.AnnotationLayer(annotations=mst)
    # )


if __name__ == '__main__':
    # INPUTS
    wormhdf5file = '/home/ashkan/workspace/myCode/MoWA/mowa-supervised-centerpred/data/train/C18G1_2L1_1.hdf'
    wormname = 'C18G1_2L1_1.hdf'
    evalfile = '/home/ashkan/workspace/myCode/MoWA/mowa-supervised-centerpred/output/snapshot/snapshot-1000.pkl'

    main(wormhdf5file, wormname, evalfile)

    print('FINISH')
