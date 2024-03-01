from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .deepro_enc import deepROEncoder
from ..parameters import Parameters


class KramerOriginal(pl.LightningModule):
    def __init__(self, args: Parameters, pretrained_cnn_path: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = args.learning_rate

        self.automatic_optimization = False

        self.mean = False

        self.encoder = deepROEncoder(2, max_pool=True)

        self._cnn_feature_vector_size = self.encoder.conv5_2[0].out_channels

        self.linear_1 = nn.Sequential(
            nn.Linear(self._cnn_feature_vector_size * 2 * 4 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(4096, 1024), nn.ReLU(inplace=True), nn.Dropout()
        )
        self.linear_3 = nn.Linear(1024, 6)

    def make_pairs(self, radar_images: List[torch.tensor]) -> torch.tensor:
        # TODO: Is torch.stack necessary? How does torch stack compare with torch cat?
        # TODO: Test with and without stack, print shape?
        tensor = torch.stack(radar_images, dim=1)
        return torch.cat([tensor[:, :-1], tensor[:, 1:]], dim=2)

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        # batch_size, sequence_length, channels = 2, height = 64, width = 128, depth = 64
        print("pairs.size() = {}".format(pairs.size()))
        batch_size, time_steps, C, H, W, D = pairs.size()
        # print('Batch_Size: ' + str(batch_size) + ' Seq Len: ' + str(time_steps) + ' Tensor Channels: '+str(C)+ ' (H,W,D)'+ '(' +str(H)+','+str(W)+','+str(D)+')')
        c_in = pairs.view(batch_size * time_steps, C, H, W, D)
        encoded_radar_images = self.encoder(c_in)
        bt_size, C_out, _, _, _ = encoded_radar_images.size()
        # print('Updated Size: ', bt_size, 'Update Channels: ', C_out, 'Full shape', encoded_radar_images.size())
        # TODO: Verify "averaging" dimensions are 2,3,4 and not 2,3 like for images channels
        # TODO: Check efficacy of mean vs max_pool vs nothing vs combination
        if self.mean:
            cnn_out = torch.mean(encoded_radar_images, dim=[2, 3, 4]).view(
                batch_size, time_steps, C_out
            )
        else:
            cnn_out = encoded_radar_images
        return cnn_out

    def forward_linear(self, encoded_images: torch.Tensor) -> torch.Tensor:
        # encoded_images = encoded_images.view(1,-1)
        time_steps, C, H, W, D = encoded_images.size()
        encoded_images = encoded_images.view(time_steps, -1)
        linear_image_1 = self.linear_1(encoded_images)
        linear_image_2 = self.linear_2(linear_image_1)
        linear_image_3 = self.linear_3(linear_image_2)
        return linear_image_3

    def forward(self, radar_images: List[torch.Tensor]) -> torch.Tensor:
        coords = self.forward_linear(self.forward_cnn(self.make_pairs(radar_images)))
        return coords

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        return [optimizer], [scheduler]

    def loss(self, gamma, y_hat, positions, orientations):
        positions = torch.stack(positions, dim=1)
        orientations = torch.stack(orientations, dim=1)

        positions = positions.squeeze()
        orientations = orientations.squeeze()

        loss_func = torch.nn.MSELoss()

        position_loss = loss_func(y_hat[:, 3:], positions)
        orientation_loss = loss_func(y_hat[:, :3], orientations)

        loss = gamma * orientation_loss + position_loss
        print(loss.shape)
        return loss

    def training_step(self, train_batch, batch_idx):
        seq_len, radar_data, positions, orientations = (
            train_batch[0],
            train_batch[1],
            train_batch[2],
            train_batch[3],
        )

        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        y_hat = self.forward(radar_data)
        loss = self.loss(1, y_hat, positions, orientations)
        self.log("Training_loss", loss)

        self.manual_backward(loss)
        opt.step()

        sch.step()
        return loss

    def validation_step(self, val_batch, batch_idx):
        seq_len, radar_data, positions, orientations = (
            val_batch[0],
            val_batch[1],
            val_batch[2],
            val_batch[3],
        )

        y_hat = self.forward(radar_data)
        loss = self.loss(1, y_hat, positions, orientations)
        self.log("Validation_loss", loss)
