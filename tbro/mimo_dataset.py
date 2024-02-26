"""
Povides custom pytorch data class for TBRO ROS node
"""

import logging
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)


class MimoDataset(Dataset):
    def __init__(self):
        super(MimoDataset, self).__init__()
        self.image_list = []

    def load_img(self, new_image):
        logging.debug("mimo:loading new_image:size: {}".format(len(new_image)))
        if len(self.image_list) > 1:
            self.image_list.pop(0)
        self.image_list.append(new_image)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index):
        image = torch.FloatTensor(self.image_list[index])
        lable = ...

        return image, lable
