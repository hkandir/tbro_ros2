from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

# from scripts.models.deepro_enc import deepROEncoder
from scripts.models.deepro_dec import deepRODecoder
from scripts.utils import weight_init
from scripts.utils.params import Parameters
from scripts.utils.losses import traj_and_odom_loss

class DeepROOriginal(pl.LightningModule):
    def __init__(
        self, args: Parameters
    ):
        super().__init__()

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.seq_length = args.max_length

        # self.encoder = deepROEncoder(2, max_pool = True)     
        
        self.lstm = nn.LSTM(
            input_size=int(2*4*2*256),
            hidden_size=args.hidden_size,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )

        self._decoder = deepRODecoder(args.hidden_size)
        self.decoder = nn.Sequential(
            nn.Dropout(),
            self._decoder
        )

        self.loss_func = traj_and_odom_loss(args.alphas, args.betas, args.gammas)

        # self.apply(weight_init)

        # if args.pr is not None:
            # self.encoder.load_pretrained(pretrained_cnn_path)


    # def make_pairs(self, radar_images: torch.tensor)-> torch.Tensor:
    #     # concatenates radar images from 0->end-1 with radar images 1->end along feature dimension (2)
    #     tensor = torch.stack(radar_images, dim=1)
    #     return torch.cat([tensor[:,:-1], tensor[:,1:]],dim=2)

    # def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
    #     # batch_size, sequence_length, channels = 2, height = 64, width = 128, depth = 64
    #     batch_size, time_steps, C, H, W, D = pairs.size()
    #     c_in = pairs.view(batch_size*time_steps, C, H, W, D)
    #     encoded_radar_images = self.encoder(c_in)
    #     bt_size, C_out, _, _ = encoded_images.size()
    #     # TODO: Verify "averaging" dimensions are 2,3,4 and not 2,3 like for images channels
    #     # TODO: Check efficacy of mean vs max_pool vs nothing vs combination
    #     if self.mean:
    #         cnn_out = torch.mean(encoded_images, dim=[2,3,4]).view(batch_size, time_steps, C_out)
    #     else:
    #         cnn_out = encoded_images
    #     return cnn_out
    
    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        return out

    def decode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.decoder(seq)

    
    def forward(self, radar_images: List[torch.Tensor]) -> torch.Tensor:
        coords = self.decode(self.forward_seq((radar_images)))
        return coords
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self,train_batch,batch_idx):
        seq_len, radar_data, positions, orientations = train_batch[0], train_batch[1], train_batch[2], train_batch[3]

        # opt = self.optimizers()
        # sch = self.lr_schedulers()
        # opt.zero_grad()

        radar_data = torch.stack(radar_data, dim=1)
        radar_data = radar_data.view(self.batch_size,self.seq_length,-1)  
        y_hat = self.forward(radar_data)
        loss = self.loss_func(y_hat,positions,orientations)
        self.log('Training_loss',loss,sync_dist=True)

        # self.manual_backward(loss)
        # opt.step()

        # sch.step()
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        seq_len, radar_data, positions, orientations = val_batch[0], val_batch[1], val_batch[2], val_batch[3]

        radar_data = torch.stack(radar_data, dim=1)
        radar_data = radar_data.view(self.batch_size,self.seq_length,-1)        
        y_hat = self.forward(radar_data)
        loss = self.loss_func(y_hat,positions,orientations)
        self.log('Validation_loss',loss,sync_dist=True)

        outputs = {"loss": loss, 
                "y_hat": y_hat,
                "positions": positions, 
                "orientations": orientations
                }


        # for key in self.visualization.keys():
        #     print(self.visualization[key][0].size())
        #     print(self.visualization[key][1].size())
        return outputs