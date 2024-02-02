import os

# from scripts.models.deepro_transformer import DeepROTransformer
# from scripts.utils.dataset import DeepRODataModule

from scripts.models.deepro_transformer_enc import DeepROTransformer
from scripts.utils.enc_dataset import DeepRODataModule

from scripts.utils.params import Parameters
from scripts.utils.losses import traj_and_odom_loss
from scripts.utils.csv_export import export_tensor_to_csv

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
        print("Callback Initialized")
        self.args = args
        
        ########################## Options #####################################
        self.basic_statistics = True
        self.export = False
        self.full_traj = True
        self.sub_traj = False
        self.one_D_error = True
        ####################### Frame to Frame Storage ########################
        self.position_gt = None
        self.orientation_gt = None

        self.position_pred = None
        self.orientation_pred = None

        self.position_err = None
        self.orientation_err = None

        self.test_loss = []
        self.val_outputs = []
        self.full_errors = full_errors

        ################################ Traj Storage #########################
        self.traj_func = traj_and_odom_loss()
        self.traj_error = None
        self.traj_gt = None
        self.traj_est = None


        ############################### Full Traj Storage ######################
        self.full_traj_error = None
        self.full_traj_gt = None
        self.full_traj_est = None

        plt.ion()
    
    def on_test_epoch_start(self, trainer, pl_module):
        print("Test Epoch Started")
        self.position_err = None
        self.orientation_err = None
        self.traj_gt = None
        self.traj_est = None
        print("Position Error, Orientation Error, Trajectoried Cleared")
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # stack outputs just in case
        self.val_outputs.append(outputs)
        
        # ------------------------------------ Frame to Frame Error ------------------------------------ #
        # generate position and orientation data in same format as loss calculations
        odom_positions = torch.stack(outputs["positions"], dim=1)
        odom_orientations = torch.stack(outputs["orientations"], dim=1)

        # print(odom_positions.shape)
        # print(outputs["y_hat"][:,:,3:].shape)
        if odom_positions.shape == outputs["y_hat"][:,:,3:].shape:
            positions = odom_positions
            orientations = odom_orientations
            # print("Same Shape in Test Data:")
        else:
            positions = odom_positions.squeeze()
            orientations = odom_orientations.squeeze()

        # calculate absolute errors and store
        position_err_batch = (outputs["y_hat"][:,:,3:] - positions).abs().view(-1,3)
        orientation_err_batch = (outputs["y_hat"][:,:,:3] - orientations).abs().view(-1,3)

        # Store Error
        if self.orientation_err is None:
            self.orientation_err = orientation_err_batch
        else:
            self.orientation_err = torch.cat((self.orientation_err, orientation_err_batch), 0)
        
        if self.position_err is None:
            self.position_err = position_err_batch
        else:
            self.position_err = torch.cat((self.position_err, position_err_batch), 0)

        # Store GT
        if self.orientation_gt is None:
            self.orientation_gt = orientations
        else:
            self.orientation_gt = torch.cat((self.orientation_gt,orientations))

        if self.position_gt is None:
            self.position_gt = positions
        else:
            self.position_gt = torch.cat((self.position_gt,positions))

        # Store Pred
        if self.orientation_pred is None:
            self.orientation_pred = outputs["y_hat"][:,:,:3]
        else:
            self.orientation_pred = torch.cat((self.orientation_pred,outputs["y_hat"][:,:,:3]))
        if self.position_pred is None:
            self.position_pred = outputs["y_hat"][:,:,3:]
        else:
            self.position_pred = torch.cat((self.position_pred,outputs["y_hat"][:,:,3:]))

        

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

    def on_test_epoch_end(self, trainer, pl_module):
        if self.basic_statistics:
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

        if self.export:
            export_tensor_to_csv(self.orientation_gt,'rot','gt_')
            export_tensor_to_csv(self.position_gt,'trans','gt_')
            export_tensor_to_csv(self.orientation_pred,'rot','pred_')
            export_tensor_to_csv(self.position_pred,'trans','pred_')

        if self.sub_traj:
            for i in range(0,self.traj_gt.size()[0]):
                x = self.traj_gt[i,0:self.args.max_length,0].squeeze()
                y = self.traj_gt[i,0:self.args.max_length,1].squeeze()
                x = x.cpu().numpy()
                y = y.cpu().numpy()

                x_hat = self.traj_est[i,0:self.args.max_length,0].squeeze()
                y_hat = self.traj_est[i,0:self.args.max_length,1].squeeze()
                x_hat = x_hat.cpu().numpy()
                y_hat = y_hat.cpu().numpy()
                
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.plot(x,y,label='Groundtruth')
                ax.plot(x_hat,y_hat,label='Predicted',dashes=[6,2])
                ax.grid(True)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Subtrajectory ' + i.__str__())
                ax.legend()
                plt.draw()
                # plt.pause(0.5)
                plt.show()
                wandb.log({"xy_plot": plt})

        if self.full_traj:
            pose_mat_gt = self.traj_func.to_transform_torch(self.position_gt,self.orientation_gt)
            pose_mat_pred = self.traj_func.to_transform_torch(self.position_pred,self.orientation_pred)
            for i in range(0,(pose_mat_gt.shape[0])):
                if(i%self.args.max_length==0):
                    if self.full_traj_gt is None:
                        self.full_traj_gt = pose_mat_gt[i,:,:,:].squeeze()
                    else: 
                        self.full_traj_gt = torch.cat((self.full_traj_gt,pose_mat_gt[i,:,:,:].squeeze()),0)
                    if self.full_traj_est is None:
                        self.full_traj_est = pose_mat_pred[i,:,:,:]
                    else:
                        self.full_traj_est = torch.cat((self.full_traj_est,pose_mat_pred[i,:,:,:].squeeze()),0)
            
            traj_gt = self.traj_func.odometry_to_track(self.full_traj_gt)
            traj_pred = self.traj_func.odometry_to_track(self.full_traj_est)

            translation_gt = traj_gt[:,:,:3,3].squeeze()
            translation_est = traj_pred[:,:,:3,3].squeeze()

            print(translation_gt.shape)

            x = translation_gt[:,0].squeeze()
            print(x.shape)
            y = translation_gt[:,1].squeeze()
            x = x.cpu().numpy()
            y = y.cpu().numpy()

            x_hat = translation_est[:,0].squeeze()
            y_hat = translation_est[:,1].squeeze()
            x_hat = x_hat.cpu().numpy()
            y_hat = y_hat.cpu().numpy()

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(x,y,label='Groundtruth')
            ax.plot(x_hat,y_hat,label='Predicted',dashes=[6,2])
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Subtrajectory ' + i.__str__())
            ax.legend()
            plt.draw()
            # plt.pause(0.5)
            plt.show()
            wandb.log({"full_traj_plot": plt})



if __name__== '__main__':
    args = Parameters()
    data_module = DeepRODataModule(args)
    data_module.setup()
    model = DeepROTransformer(args)
    wandb_logger = WandbLogger(project="TBRO", offline=True)
    # trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu',devices=2,strategy=DDPStrategy(find_unused_parameters=False),max_epochs=args.epochs)
    trainer = pl.Trainer(logger=wandb_logger, accelerator='gpu',devices=1,max_epochs=args.epochs, callbacks=[rpe_callback(args,True)],default_root_dir=os.path.join('/home/kharlow/checkpoints/4_18_23/vel_test'))
    trainer.test(model,data_module,ckpt_path = '/home/kharlow/checkpoints/4_21_23/TBRO/yvhakvgz/checkpoints/epoch=49-step=687900.ckpt')