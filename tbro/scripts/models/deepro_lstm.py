from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

from scripts.models.deepro_enc import deepROEncoder
from scripts.models.deepro_dec import deepRODecoder
from scripts.utils import weight_init
from scripts.utils.params import Parameters

class DeepROOriginal(pl.LightningModule):
    def __init__(
        self, args: Parameters, radar_shape: Tuple[int,int,int], sequence_len: int, hidden_size: int
    ):
        super().__init__()

        self.encoder = deepROEncoder(2, max_pool = True)

        encoder_output = self.encoder(torch.ones((1, sequence_len - 1, *radar_shape)))
        self._cnn_feature_vector_size = int(encoder_output.numel())        
        
        self.lstm = nn.LSTM(
            input_size=self._cnn_feature_vector_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0,
            batch_first=True
        )

        # TODO: Read and understand TBVO decoder layers
        self.decoder = nn.Sequential(
            nn.Dropout(),
            pose_decoder
        )

        # self.apply(weight_init)

        if args.pr is not None:
            self.encoder.load_pretrained(pretrained_cnn_path)


    def make_pairs(self, radar_images: torch.tensor)-> torch.Tensor:
        # concatenates radar images from 0->end-1 with radar images 1->end along feature dimension (2)
        tensor = torch.stack(radar_images, dim=1)
        return torch.cat([tensor[:,:-1], tensor[:,1:]],dim=2)

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        # batch_size, sequence_length, channels = 2, height = 64, width = 128, depth = 64
        batch_size, time_steps, C, H, W, D = pairs.size()
        c_in = pairs.view(batch_size*time_steps, C, H, W, D)
        encoded_radar_images = self.encoder(c_in)
        bt_size, C_out, _, _ = encoded_images.size()
        # TODO: Verify "averaging" dimensions are 2,3,4 and not 2,3 like for images channels
        # TODO: Check efficacy of mean vs max_pool vs nothing vs combination
        if self.mean:
            cnn_out = torch.mean(encoded_images, dim=[2,3,4]).view(batch_size, time_steps, C_out)
        else:
            cnn_out = encoded_images
        return cnn_out
    
    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return out

    def decode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.decoder(seq)

    
    def forward(self, radar_images: List[ptorch.Tensor]) -> torch.Tensor:
        coords, quats = self.decode(self.forward_seq(self.forward_cnn(self.make_pairs(images))))
        return to_homogeneous_torch(coords, quats)
    # TODO: What is an @property?
    # @property
    # def cnn_feature_vector_size(self) -> int:
    #     return self._cnn_feature_vector_size