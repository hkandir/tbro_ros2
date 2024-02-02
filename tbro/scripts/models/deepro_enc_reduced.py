import torch
import torch.nn as nn
import pytorch_lightning as pl

def _conv(
    in_channels: int, out_channels: int, kernel_size: int = 3,
    stride: int=1, padding: int=1, droupout: float = 0, batch_norm: bool = True
) -> nn.Sequential:
    layers = [nn.Conv3d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        bias = not batch_norm
    )]

    if batch_norm:
        layers.append(
            nn.BatchNorm3d(out_channels)
        )
    
    layers.append(
        #TODO: Test GELU, RELU
        nn.LeakyReLU(0.1)
    )

    layers.append(
        nn.Dropout(droupout)
    )

    #TODO: What does * mean in python?
    return nn.Sequential(*layers)

class deepROEncoder(pl.LightningModule):
    def __init__(self, in_channels: int, batch_norm: bool = True, max_pool: bool = False):
        super().__init__()

        self._batch_norm = batch_norm
        self._max_pool = max_pool
        #TODO: Test Kernel Size Differences
        self.conv1_1 = _conv(in_channels, 32, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)
        self.conv1_2 = _conv(32, 32, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)

        self.conv2_1 = _conv(32, 64, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)
        self.conv2_2 = _conv(64, 64, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)

        self.conv3_1 = _conv(64, 128, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)
        self.conv3_2 = _conv(128, 128, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)

        self.conv4_1 = _conv(128, 256, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)
        self.conv4_2 = _conv(256, 256, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)

        # self.conv5_1 = _conv(256, 256, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)
        # self.conv5_2 = _conv(256, 256, kernel_size = 3, stride = 1, padding = 1, droupout = 0.0, batch_norm = self._batch_norm)

        self._valid_conv_names = [
            f"conv{i}" for i in range(1,4)
        ]

        if max_pool:
            self.max_pool = torch.nn.MaxPool3d(2,2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._max_pool:
            out_conv1 = self.conv1_2(self.conv1_1(x))
            out_conv1 = self.max_pool(out_conv1)
            out_conv2 = self.conv2_2(self.conv2_1(out_conv1))
            out_conv2 = self.max_pool(out_conv2)
            out_conv3 = self.conv3_2(self.conv3_1(out_conv2))
            out_conv3 = self.max_pool(out_conv3)
            out_conv4 = self.conv4_2(self.conv4_1(out_conv3))
            out_conv4 = self.max_pool(out_conv4)
            # out_conv5 = self.conv5_2(self.conv5_1(out_conv4))
            # out_conv5 = self.max_pool(out_conv5)
        else: 
            out_conv1 = self.conv1_2(self.conv1_1(x))
            out_conv2 = self.conv2_2(self.conv2_1(out_conv1))
            out_conv3 = self.conv3_2(self.conv3_1(out_conv2))
            out_conv4 = self.conv4_2(self.conv4_1(out_conv3))
            # out_conv5 = self.conv5_2(self.conv5_1(out_conv4))

        return out_conv4

    # Allows us to train weights on modified network and return only weights from encoder layers
    # i.e. train kramer_original, load kramer_encoder
    def load_weights(self, path: str):
        model_dict = torch.load(path, map_location='cpu')
        # print(self.state_dict())
        self.load_state_dict(model_dict)
        # print(self.state_dict())
    #     state_dict = {
    #         i: j for i, j in model_dict['state_dict'].items() if self._is_layer_valid(i)
    #     }

    #     self.load_state_dict(state_dict)

    # def _is_layer_valid(self, state: str):
    #     if state.startswith("conv6"):
    #         print("Ignoring: "  + i)
    #         return False

    #     for i in self._valid_conv_names:
    #         if state.startswith(i):
    #             print("Loading: " + i)
    #             return True    
    
    #     return False
        



