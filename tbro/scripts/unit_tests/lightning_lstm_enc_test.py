import os

from scripts.models.deepro_lstm_enc import DeepROOriginal

from scripts.utils.enc_dataset import DeepRODataModule
from scripts.utils.params import Parameters
from scripts.utils.losses import traj_and_odom_loss

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
import torch
import wandb

import matplotlib.pyplot as plt
import numpy as np

class rpe_callback(Callback):
    def __init__(self, args: Parameters, full_errors: bool):
        self.args = args
        self.position_err = None
        self.orientation_err = None
        self.test_loss = []
        self.val_outputs = []
        self.full_errors = full_errors

        self.traj_func = traj_and_odom_loss()
        self.traj_error = None
        self.traj_gt = None
        self.traj_est = None
        plt.ion()
    
    def on_validation_epoch_start(self, trainer, pl_module):
        self.position_err = None
        self.orientation_err = None
        self.traj_gt = None
        self.traj_est = None
            
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # stack outputs just in case
        self.val_outputs.append(outputs)
        

        # ------------------------------------ Frame to Frame Error ------------------------------------ #
        # generate position and orientation data in same format as loss calculations
        odom_positions = torch.stack(outputs["positions"], dim=1)
        odom_orientations = torch.stack(outputs["orientations"], dim=1)
        positions = odom_positions.squeeze()
        orientations = odom_orientations.squeeze()

        # calculate absolute errors and store
        position_err_batch = (outputs["y_hat"][:,:,3:] - positions).abs().view(-1,3)
        orientation_err_batch = (outputs["y_hat"][:,:,:3] - orientations).abs().view(-1,3)

        if self.orientation_err is None:
            self.orientation_err = orientation_err_batch
        else:
            self.orientation_err = torch.cat((self.orientation_err, orientation_err_batch), 0)
        
        if self.position_err is None:
            self.position_err = position_err_batch
        else:
            self.position_err = torch.cat((self.position_err, position_err_batch), 0)

        # ------------------------------------ Traj  ------------------------------------ #
        pose_mat_gt = self.traj_func.to_transform_torch(positions,orientations)
        traj_gt = self.traj_func.odometry_to_track(pose_mat_gt)

        pose_mat_est = self.traj_func.to_transform_torch(outputs["y_hat"][:,:,3:],outputs["y_hat"][:,:,:3])
        traj_est = self.traj_func.odometry_to_track(pose_mat_est)

        translation_gt = traj_gt[:,:,:3,3]
        translation_est = traj_est[:,:,:3,3]


        if self.traj_gt is None:
            self.traj_gt = translation_gt
        else:
            self.traj_gt = torch.cat((self.traj_gt,translation_gt),0)
        if self.traj_est is None:
            self.traj_est = translation_est
        else:
            self.traj_est = torch.cat((self.traj_est,translation_est),0)              

    def on_validation_epoch_end(self, trainer, pl_module):
        print("Frame-to-Frame Error Statistics")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        or_err_mean = self.orientation_err.mean(dim=0)
        or_err_max = self.orientation_err.max(dim=0).values
        or_err_min = self.orientation_err.min(dim=0).values
        or_err_median = self.orientation_err.median(dim=0).values
        or_err_std_dev = self.orientation_err.std(dim=0)

        pos_err_mean = self.position_err.mean(dim=0)
        pos_err_max = self.position_err.max(dim=0).values
        pos_err_min = self.position_err.min(dim=0).values
        pos_err_median = self.position_err.median(dim=0).values
        pos_err_std_dev = self.position_err.std(dim=0)

        self.log_dict({"mean_rel_x_err": pos_err_mean[0],
                  "mean_rel_y_err": pos_err_mean[1],
                  "mean_rel_z_err": pos_err_mean[2]})
        self.log_dict({"mean_rel_r_err": or_err_mean[0],
                  "mean_rel_p_err": or_err_mean[1],
                  "mean_rel_q_err": or_err_mean[2]})

        print('\tMean orientation error (rpy, rad) - %f, %f, %f' % 
            (or_err_mean[0],
            or_err_mean[1],
            or_err_mean[2]))

        if self.full_errors:
            print('\tMedian orientation error (rpy, rad) - %f, %f, %f' % 
            (or_err_median[0],
            or_err_median[1],
            or_err_median[2]))

            print('\tMax orientation error (rpy, rad) - %f, %f, %f' %
            (or_err_max[0],
            or_err_max[1],
            or_err_max[2]))

            print('\tMin orientation error (rpy, rad) - %f, %f, %f' %
            (or_err_min[0],
            or_err_min[1],
            or_err_min[2]))

            print('\tOrientation error std dev (rpy, rad) - %f, %f, %f\n' %
            (or_err_std_dev[0],
            or_err_std_dev[1],
            or_err_std_dev[2]))

        print('\tMean position error (xyz, m) - %f, %f, %f' % 
            (pos_err_mean[0],
            pos_err_mean[1],
            pos_err_mean[2]))

        if self.full_errors: 
            print('\tMedian position error (xyz, m) - %f, %f, %f' % 
            (pos_err_median[0],
            pos_err_median[1],
            pos_err_median[2]))

            print('\tMax position error (xyz, m) - %f, %f, %f' %
            (pos_err_max[0],
            pos_err_max[1],
            pos_err_max[2]))

            print('\tMin position error (xyz, m) - %f, %f, %f' %
            (pos_err_min[0],
            pos_err_min[1],
            pos_err_min[2]))

            print('\tPosition error std dev (xyz, m) - %f, %f, %f' %
            (pos_err_std_dev[0],
            pos_err_std_dev[1],
            pos_err_std_dev[2]))

        # print('\n')
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # first batch, first sequence, x or y
        # print(self.traj_gt.size())
        
        if self.traj_gt.size()[0] >= 100:
            x = self.traj_gt[100,0:self.args.max_length,0].squeeze()
            y = self.traj_gt[100,0:self.args.max_length,1].squeeze()
            x = x.cpu()
            y = y.cpu()
            x = x.numpy()
            y = y.numpy()

            x_hat = self.traj_est[100,0:self.args.max_length,0].squeeze()
            y_hat = self.traj_est[100,0:self.args.max_length,1].squeeze()
            x_hat = x_hat.cpu()
            y_hat = y_hat.cpu()
            x_hat = x_hat.numpy()
            y_hat = y_hat.numpy()
            
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(x,y)
            ax.plot(x_hat,y_hat,dashes=[6,2])
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.draw()
            plt.pause(2)
            plt.show()
            wandb.log({"xy_plot": plt})




if __name__== '__main__':
    args = Parameters()
    data_module = DeepRODataModule(args)
    data_module.setup()
    model = DeepROOriginal(args)
    wandb_logger = WandbLogger(project="TBRO", offline=True)
    # trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu',devices=1,strategy=DDPStrategy(find_unused_parameters=False),max_epochs=args.epochs)
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu',devices=1,max_epochs=args.epochs, callbacks=[rpe_callback(args,True)])
    trainer.fit(model,data_module)
