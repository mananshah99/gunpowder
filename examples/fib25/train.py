from __future__ import print_function

import os

import h5py
import malis
import math

import gunpowder
from gunpowder import VolumeType, RandomLocation, Reject, Normalize, RandomProvider, GrowBoundary, \
    SplitAndRenumberSegmentationLabels, AddGtAffinities, PreCache, Snapshot, BatchRequest, ElasticAugment, \
    SimpleAugment, IntensityAugment, IntensityScaleShift
from gunpowder.caffe import Train
from gunpowder.nodes.dvid_source import DvidSource

import constants


def train():

    gunpowder.set_verbose()

    affinity_neighborhood = malis.mknhood3d()
    solver_parameters = gunpowder.caffe.SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 1e-4
    solver_parameters.momentum = 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 10000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    request = BatchRequest()
    request.add_volume_request(VolumeType.RAW, constants.input_shape)
    request.add_volume_request(VolumeType.GT_LABELS, constants.output_shape)
    # request.add_volume_request(VolumeType.GT_MASK, constants.output_shape)
    # request.add_volume_request(VolumeType.GT_IGNORE, constants.output_shape)
    request.add_volume_request(VolumeType.GT_AFFINITIES, constants.output_shape)


    data_sources = list()
    for volume_name, path in {'tstvol-520-1-h5': '/home/ubuntu/data/FlyEM/fibsem_medulla_7col/tstvol-520-1-h5/'}.iteritems():
        h5_filepath = "./{}.h5".format(volume_name)
        print(h5_filepath)
        with h5py.File(h5_filepath, "w") as h5:
            h5['volumes/raw'] = h5py.ExternalLink(os.path.join(path, "img_normalized.h5"), "main")
            h5['volumes/labels/neuron_ids'] = h5py.ExternalLink(os.path.join(path, "groundtruth_seg.h5"), "main")
        data_sources.append(
            gunpowder.Hdf5Source(
                h5_filepath,
                datasets={
                    VolumeType.RAW: 'volumes/raw',
                    VolumeType.GT_LABELS: 'volumes/labels/neuron_ids',
                },
                resolution=(8, 8, 8),
            )
        )
    data_sources = tuple(
        data_source + \
        RandomLocation() + \
        # Reject() + \
        Normalize(factor=(0.5 ** 8))
        for data_source in data_sources
    )

    # snapshot_request = BatchRequest()
    # snapshot_request.add_volume_request(VolumeType.LOSS_GRADIENT, constants.output_shape)

    snapshot_request = request

    # create a batch provider by concatenation of filters
    batch_provider = (
        data_sources +
        RandomProvider() +
        ElasticAugment([1, 1, 1], [1, 1, 1], [0, math.pi / 2.0], prob_slip=0.05, prob_shift=0.05, max_misalign=25) +
        SimpleAugment(transpose_only_xy=False) +
        GrowBoundary(steps=2, only_xy=False) +
        AddGtAffinities(affinity_neighborhood) +
        SplitAndRenumberSegmentationLabels() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=False) +
        IntensityScaleShift(2, -1) +
        PreCache(
            request,
            cache_size=2,
            num_workers=1) +
        Train(solver_parameters, use_gpu=0) +
        Snapshot(every=1, output_filename='batch_{id}.hdf', additional_request=snapshot_request)
    )

    n = 10
    print("Training for", n, "iterations")

    with gunpowder.build(batch_provider) as pipeline:
        for i in range(n):
            pipeline.request_batch(request)

    print("Finished")


if __name__ == "__main__":
    train()
