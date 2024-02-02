import torch
import torch.nn as nn
import pytorch_lightning as pl
from scripts.utils.params import Parameters


_valid_conv_names = [
            f"encoder.conv{i}" for i in range(1,6)
        ]

def _is_layer_valid(state: str):
    if state.startswith("linear"):
        print("Ignoring linear layer")
        return False

    for i in _valid_conv_names:
        if state.startswith(i):
            print("Loading: " + i)
            return True    

    return False

# def remove_enc_prefix(key):


# Allows us to train weights on modified network and return only weights from encoder layers
# i.e. train kramer_original, load kramer_encoder
def load_weights(path: str):
    model_dict = torch.load(path, map_location='cpu')
    # print(model_dict)
    print(model_dict.keys())
    state_dict = {
        i.removeprefix('encoder.'): j for i, j in model_dict.items() if _is_layer_valid(i)
    }
    print(state_dict.keys())
    torch.save(state_dict,'/home/arpg/results_training/results/best_encoder_modified.model')
    # state_dict = {
    #     i: j for i, j in model_dict.items() if self._is_layer_valid(i)
    # }
    # test_dict = self.state_dict()
    # self.load_state_dict(model_dict, strict=False)
    # test_dict_2 = self.state_dict()
    # print("Dictionary is equal?: " + (test_dict == test_dict_2).__str__)


if __name__ == '__main__':
    args = Parameters()
    path = args.pretrained_enc_path
    load_weights(path)