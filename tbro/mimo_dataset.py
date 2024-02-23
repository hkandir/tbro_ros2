from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MimoDataset(Dataset):
    def __init__(self):
        super(MimoDataset, self).__init__()
        self.image_list = []

    def load_img(self, new_image):
        if len(self.image_list) > 1:
            self.image_list.pop(0)
        self.image_list.append(new_image)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.transform(self.image_list[index])
        lable = ...

        return image, lable
