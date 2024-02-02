import math
import numpy as np

import torch
import torch.nn  as nn
from scripts.utils.dataset import DeepRODataset
from scripts.utils.params import Parameters
from torch.utils.data import DataLoader


import scipy.spatial.transform as tf 

def to_transform_torch(positions: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Convert lists of coords and quaternions to matrix poses."""
    batch_size = positions.shape[0]
    seq_len = positions.shape[1]


    poses_mat = torch.zeros(batch_size, seq_len, 4, 4, dtype=gt_track_positions.dtype, device=gt_track_positions.device)
    # print(poses_mat.shape)
    # print(gt_track_orientations.dtype)
    # print(gt_track_orientations[0,0,:])
    # for i,j in enumerate(gt_track_orientations):
    #     print(i)
    #     for k,h in enumerate(j):
    #         print(h)



    poses_mat[:, :, 3, 3] += 1
    poses_mat[:, :, :3, 3] += gt_track_positions
    for i,j in enumerate(gt_track_orientations):
        for k,h in enumerate(j):
            # if i<1 and k<1:
                # print(gt_track_positions[i,k])
                # print(gt_track_orientations[i,k])
                # Rot_mat = tf.Rotation.from_euler('xyz',h.tolist())
                # poses_mat[i, k, :3, :3] += Rot_mat.as_matrix()
                # print(poses_mat[i,k,:,:])
            rot_mat = tf.Rotation.from_euler('xyz',h.tolist())
            # print(Rot_mat.shape())
            # print(i,k)
            # print(rot_mat.as_matrix())
            poses_mat[i, k, :3, :3] += rot_mat.as_matrix()
            # print(poses_mat)
    # print(poses_mat.shape)
    return poses_mat

def odomtery_to_track(poses_mat: torch.Tensor) -> torch.Tensor:
    # Shape (batch_size, sequence_length, 6dof (xyz,rpq))
    batch_size = poses_mat.shape[0]
    seq_len = poses_mat.shape[1]

    first_pose = torch.tile(torch.eye(4, dtype=poses_mat.dtype, device=poses_mat.device).unsqueeze(0), (batch_size, 1, 1))

    track = [first_pose]
    start_index = 1

    for i in range(start_index, seq_len):
        pose = poses_mat[:,i]
        prev = track[-1]

        track.append(torch.matmul(prev,pose))

    track = torch.stack(track, dim=1)

    return track


if __name__ == '__main__':
    args = Parameters()
    test_dataset = DeepRODataset(root_dir = args.directory, list_files = args.val_files, verbose_debug = False, sequence_length = args.max_length)
    print('Loaded ' + str(len(test_dataset)) + ' testing sequences.')
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)


    for i, data in enumerate(test_loader):
        seq_len, radar_data, positions, orientations = data[0], data[1], data[2], data[3]
        # traj = odomtery2track(positions,orientations)
        gt_track_positions = torch.stack(positions, dim=1)
        gt_track_orientations = torch.stack(orientations, dim=1)
        gt_track_positions = gt_track_positions.squeeze()
        gt_track_orientations = gt_track_orientations.squeeze()
        
        # print(gt_track_positions.shape)
        # print(gt_track_orientations.shape)
        poses_mat = to_homogeneous_torch(gt_track_positions,gt_track_orientations)
        # print(gt_track_positions[0,0,:])
        # print(gt_track_orientations[0,0,:])
        # print(poses_mat[0,0,:,:])
        if i<1:
            track = odomtery2track(poses_mat)
            print(poses_mat)
            print(poses_mat.shape)
            print(track)
            print(track.shape)
    