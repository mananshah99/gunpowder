from __future__ import print_function

import argparse
import glob
import time

import dvision
import h5py
import neuroglancer
import numpy as np
from neuroglancer.base_viewer import Layer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('snapshot_filepath', type=str, default='',
                    help='path to hdf5 file containing snapshot arrays')
args = parser.parse_args()

snapshots = list(sorted(glob.glob("./fib25/snapshots/*")))
snapshot_filepath = args.snapshot_filepath or snapshots[0]
print(snapshot_filepath)

neuroglancer.server.global_server_args["bind_address"] = '0.0.0.0'
neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')

three_channel_float_shader = """
    void main() {
      emitRGB(vec3(toNormalized(getDataValue(0)),
                   toNormalized(getDataValue(1)),
                   toNormalized(getDataValue(2))));
    }
"""


def get_nanometer_xyz_offset(zyx_voxel_offset, xyz_resolution):
    return tuple(vo * r for vo, r in zip(reversed(zyx_voxel_offset[-3:]), xyz_resolution))

source_datasets = {
    "/volumes/raw": dvision.DVIDDataInstance("slowpoke3", 32788, "341", "grayscale"),
    "/volumes/labels/neuron_ids": dvision.DVIDDataInstance("slowpoke3", 32788, "341", "groundtruth_pruned")
}

shaders = {
    # "/volumes/predicted_affs": three_channel_float_shader,
    # "/volumes/gt_affs": three_channel_float_shader
}

snapshot_dataset_keys = [
    "/volumes/raw",
    "/volumes/labels/neuron_ids",
    "/volumes/labels/mask",
    "/volumes/gt_affs",
    "/volumes/predicted_affs",
]

dtypes = {
    "/volumes/labels/neuron_ids": np.dtype("uint32"),
    "/volumes/labels/mask": np.dtype("uint8")
}

viewer = neuroglancer.Viewer(voxel_size=(8, 8, 8))

with h5py.File(snapshot_filepath, "r") as f:
    for key in snapshot_dataset_keys:
        dataset = f[key]
        zyx_voxel_offset = dataset.attrs['offset']
        zyx_resolution = dataset.attrs['resolution']
        xyz_resolution = list(reversed(zyx_resolution))
        array = np.array(dataset).squeeze()
        dtype = dtypes.get(key, np.dtype("float32"))
        if array.dtype is not dtype:
            print("recasting", key, "from", array.dtype, "to", dtype)
            new_array = array.astype(dtype)
            np.testing.assert_array_equal(array, new_array)
            array = new_array
        print(key, array.shape, array.dtype, zyx_voxel_offset)
        xyz_nanometer_offset = get_nanometer_xyz_offset(zyx_voxel_offset, xyz_resolution)
        print(xyz_nanometer_offset)
        viewer.add(
            array,
            name=key,
            offset=xyz_nanometer_offset,
            voxel_size=xyz_resolution,
            shader=shaders.get(key, None)
        )

for key, dvid_dataset in source_datasets.items():
    viewer.add(
        dvid_dataset,
        name="dvid - {}".format(key),
        offset=(0, 0, 0),
        voxel_size=(8, 8, 8),
    )


# 'voxelCoordinates':[2703_2265_4128]

print(viewer)

while True:
    time.sleep(100)
    pass  # don't exit









