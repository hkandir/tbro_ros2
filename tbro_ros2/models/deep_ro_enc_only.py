"""
TBRO mode discription class
"""

# generidc
from typing import List


# pytorch
import torch

# local imports
from .parameters import Parameters
from .models.deepro_enc import deepROEncoder


class DeepROEncOnly(torch.nn.Module):
    def __init__(self, args: Parameters):
        super().__init__()

        self.save_hyperparameters = {
            "sequence_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "radar_shape": args.radar_shape,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "alphas": args.alphas,
            "betas": args.betas,
            "gammas": args.gammas,
        }
        self.learning_rate = args.learning_rate

        # self.automatic_optimization = False
        # self.loss_func = traj_and_odom_loss(args.alphas, args.betas, args.gammas)

        self.mean = args.mean_enable
        self.learning_rate = args.learning_rate

        self.encoder = deepROEncoder(
            in_channels=2, max_pool=True
        )  # Two radar images (power channel only)
        self._cnn_feature_vector_size = self.encoder.conv5_2[0].out_channels

        if args.pretrained_enc_path is not None:
            self.encoder.load_weights(args.pretrained_enc_path)

    def make_pairs(self, radar_images: List[torch.Tensor]) -> torch.Tensor:
        tensor = torch.stack(radar_images, dim=1)
        return torch.cat([tensor[:, :-1], tensor[:, 1:]], dim=2)

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        # batch_size, sequence_length, channels = 2, height = 64, width = 128, depth = 64
        batch_size, time_steps, C, H, W, D = pairs.size()
        c_in = pairs.view(batch_size * time_steps, C, H, W, D)
        encoded_radar_images = self.encoder(c_in)
        bt_size, C_out, _, _, _ = encoded_radar_images.size()
        if self.mean:
            cnn_out = torch.mean(encoded_radar_images, dim=[2, 3, 4]).view(
                batch_size, time_steps, C_out
            )
        else:
            cnn_out = encoded_radar_images
        return cnn_out

    def forward(self, radar_images: List[torch.Tensor]) -> torch.Tensor:
        encoded_images = self.forward_cnn(self.make_pairs(radar_images))
        return encoded_images

    def training_step(self, train_batch, batch_idx):
        seq_len, radar_data, positions, orientations = (
            train_batch[0],
            train_batch[1],
            train_batch[2],
            train_batch[3],
        )

        # opt = self.optimizers()
        # sch = self.lr_schedulers()
        # opt.zero_grad()

        y_hat = self.forward(radar_data)

        # torch.save([y_hat,positions,orientations],self.args.directory + 'test/')

        # loss = self.loss_func(y_hat,positions,orientations)
        # self.log('Training_loss',loss,sync_dist=True)

        # self.manual_backward(loss)
        # opt.step()

        # sch.step()
        # return loss

    def validation_step(self, val_batch, batch_idx):
        seq_len, radar_data, positions, orientations = (
            val_batch[0],
            val_batch[1],
            val_batch[2],
            val_batch[3],
        )

        y_hat = self.forward(radar_data)
        # loss = self.loss_func(y_hat,positions,orientations)
        # self.log('Validation_loss',loss,sync_dist=True)
        # return loss
