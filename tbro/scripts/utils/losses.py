import torch
import torch.nn as nn

import math
import numpy as np

import scipy.spatial.transform as tf 
import wandb

class traj_and_odom_loss(nn.Module):
    def __init__(self, alphas = None, betas = None, gammas=None):
        super().__init__()

        # Learning Rates
        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
               
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        

    def to_transform_torch(self, positions: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
        batch_size = positions.shape[0]
        seq_len = positions.shape[1]

        poses_mat = torch.zeros(batch_size, seq_len, 4, 4, dtype=positions.dtype, device=positions.device)
      
        poses_mat[:, :, 3, 3] += 1
        poses_mat[:, :, :3, 3] += positions
        for i,j in enumerate(orientations):
            for k,h in enumerate(j):
                rot_mat = tf.Rotation.from_euler('xyz',h.tolist())
                poses_mat[i, k, :3, :3] += torch.tensor(rot_mat.as_matrix(), device=positions.device)
        return poses_mat
    
    def odometry_to_track(self, poses_mat: torch.Tensor) -> torch.Tensor:
        # Shape (batch_size, sequence_length, 6dof (xyz,rpq))
        if len(poses_mat.shape) == 4:
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
        if len(poses_mat.shape) == 3:
            print('Track without batching')
            seq_len = poses_mat.shape[0]

            first_pose = torch.tile(torch.eye(4, dtype=poses_mat.dtype, device=poses_mat.device).unsqueeze(0), (1, 1))

            track = [first_pose]
            start_index = 1

            for i in range(start_index, seq_len):
                pose = poses_mat[i]
                prev = track[-1]

                track.append(torch.matmul(prev,pose))

            track = torch.stack(track, dim=1)
            print(track.shape)
        return track

    def rotation_loss(self, rot_1: torch.Tensor, rot_2: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-6
        R = torch.matmul(rot_1,rot_2.transpose(2,3))
        # (tr(R)-1)/2
        cos = (R[:,:,0,0] + R[:,:,1,1] + R[:,:,2,2] - 1.)/2.
        # gaurantee arccos will not fail
        cos = torch.clamp(cos, -1 + epsilon, 1-epsilon)
        theta = torch.acos(cos)
        return torch.mean(theta)

    def odometry_loss(self,pred,gt_odom_positions,gt_odom_orientations):
        mse_position_loss = self.mse_loss(pred[:,:,3:], gt_odom_positions)
        mae_position_loss = self.mae_loss(pred[:,:,3:], gt_odom_positions)

        pose_mat_est = self.to_transform_torch(pred[:,:,3:],pred[:,:,:3])
        pose_mat_gt = self.to_transform_torch(gt_odom_positions,gt_odom_orientations)
        orientation_loss = self.rotation_loss(pose_mat_est[:,:,:3,:3],pose_mat_gt[:,:,:3,:3])


        # mse_orientation_loss = self.mse_loss(pred[:,:,:3], gt_odom_orientations)
        # mae_orientation_loss = self.mae_loss(pred[:,:,:3], gt_odom_orientations)

        position_loss = mse_position_loss + mae_position_loss
        # orientation_loss = mse_orientation_loss + mae_orientation_loss

        wandb.log({"mse_odom_trans_loss": mse_position_loss,
                    "mae_odom_trans_loss": mae_position_loss,
                    "odom_rot_loss": orientation_loss})

        odom_loss = self.alphas[0] * orientation_loss + self.alphas[1] * position_loss
        return odom_loss

    def trajectory_loss(self,pred,gt_odom_positions,gt_odom_orientations):
        pose_mat_gt = self.to_transform_torch(gt_odom_positions,gt_odom_orientations)
        traj_gt = self.odometry_to_track(pose_mat_gt)

        pose_mat_est = self.to_transform_torch(pred[:,:,3:],pred[:,:,:3])
        traj_est = self.odometry_to_track(pose_mat_est)

        translation_gt = traj_gt[:,:,:3,3]
        translation_est = traj_est[:,:,:3,3]

        mse_position_loss = self.mse_loss(translation_est,translation_gt)
        mae_position_loss = self.mae_loss(translation_est,translation_gt)
        position_loss = mse_position_loss + mae_position_loss
     
        orientation_loss = self.rotation_loss(traj_est[:,:,:3,:3],traj_gt[:,:,:3,:3])

        wandb.log({"mse_traj_trans_loss": mse_position_loss,
                    "mae_traj_trans_loss": mae_position_loss,
                    "traj_rot_loss": orientation_loss})
         
        traj_loss = self.betas[0] * orientation_loss + self.betas[1] * position_loss
        return traj_loss

    def forward(self, y_hat, positions, orientations):
        # Predicted values
        pred = y_hat

        # Known values
        odom_positions = torch.stack(positions, dim=1)
        odom_orientations = torch.stack(orientations, dim=1)
        if odom_positions.shape == y_hat[:,:,3:].shape:
            # print("Same Shape")
            gt_odom_positions = odom_positions
            gt_odom_orientations = odom_orientations
        else:
            gt_odom_positions = odom_positions.squeeze()
            gt_odom_orientations = odom_orientations.squeeze()

        if self.alphas is not None:
            odom_loss = self.odometry_loss(pred,gt_odom_positions,gt_odom_orientations)
        else:
            odom_loss = 0

        if self.betas is not None:
            traj_loss = self.trajectory_loss(pred,gt_odom_positions,gt_odom_orientations)
        else:
            traj_loss = 0


        self.loss = self.gammas[0]*odom_loss + self.gammas[1]*traj_loss
        # print("Odom: ", odom_loss)
        # print("Traj: ", traj_loss)
        return self.loss
