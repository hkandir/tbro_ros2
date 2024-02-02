import os
import shutil

import torch

from scripts.utils.enc_dataset import DeepRODataset as e_data
from scripts.utils.dataset import DeepRODataset as data
from scripts.utils.params import Parameters

from torch.utils.data import DataLoader
import numpy as np

if __name__ == '__main__':
    args = Parameters()
    print("Root dir: " + args.directory)
    print("list: " + args.train_files)

    e_dataset = e_data(root_dir = args.directory, list_files = args.train_files, verbose_debug = False, sequence_length = args.max_length)
    print('Loaded ' + str(len(e_dataset)) + ' encoded sequences.')
    enc_loader = DataLoader(e_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,drop_last=True)

    dataset = data(root_dir = args.directory, list_files = args.train_files, verbose_debug = False, sequence_length = args.max_length)
    print('Loaded ' + str(len(dataset)) + ' regular sequences.')
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,drop_last=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        print('CUDA used.')
    else:
        device = torch.device('cpu')
        print('CPU used.')

    for i in range(1):
        seq_len, radar_data, positions, orientations = e_dataset.__getitem__(i)
        print('Position Length: ',len(positions))
        print('Orientation Length: ',len(orientations))
        positions = torch.stack(positions, dim=1)
        orientations = torch.stack(orientations, dim=1)

        positions = positions.squeeze()
        orientations = orientations.squeeze()
        
        # print(positions.size())
        # print(orientations.size())
        if torch.any(torch.isnan(positions)):
            print("Nan found in positions batch:",i)
        if torch.any(torch.isnan(orientations)):
            print("Nan found in orientations batch:",i)
        positions = positions.to(device)
        orientations = orientations.to(device)
        if positions.size(dim=0) != 3:
            print(positions.size())
        if positions.size(dim=1) != seq_len:
            print(positions.size())
        if orientations.size(dim=0) != 3:
            print(orientations.size())
        if orientations.size(dim=1) != seq_len:
            print(orientations.size())

        
        # for j in range(args.batch_size):
        print('Radar Sequence Length: ', len(radar_data))
        print('Expected Sequence Length: ', seq_len)
        for k in range(seq_len):
            # print(k)
            # print(radar_data[k].size())
            radar_data[k] = radar_data[k].to(device)
        # if radar_data[k].size(dim=0) != 256:
        #     print(radar_data[k].size())
        if torch.any(torch.isnan(radar_data[k])):
            print('Image (item,seq_idx)'+str(i,k)+' contains NaN')
        
        if seq_len != args.max_length:
            print(seq_len,i)


        # seq_len, radar_data, positions, orientations = dataset.__getitem__(i)
        # positions = torch.stack(positions, dim=1)
        # orientations = torch.stack(orientations, dim=1)

        # positions = positions.squeeze()
        # orientations = orientations.squeeze()
        
        # print(positions.size())
        # print(orientations.size())
        # print(len(radar_data))
        # for k in range(seq_len):
        #     print(radar_data[k].size())


    for i,data in enumerate(enc_loader):
        seq_len, radar_data, positions, orientations = data[0], data[1], data[2], data[3]
        positions = torch.stack(positions, dim=1)
        orientations = torch.stack(orientations, dim=1)

        positions = positions.squeeze()
        orientations = orientations.squeeze()
        

        # print(positions.size())
        # # print(len(radar_data))
        # for i in range(len(radar_data)):
        #     print(radar_data[i].size())

        radar_data = torch.stack(radar_data,dim=1)
        print(radar_data.size())        

#     args = Parameters()
#     print("Root dir: " + args.directory)
#     print("list: " + args.train_files)
#     e_dataset = e_data(root_dir = args.directory, list_files = args.train_files, verbose_debug = False, sequence_length = args.max_length)
#     # print("Temp Dataset: ", e_dataset)
#     enc_loader = DataLoader(e_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)

#     print("Temp Dataset __len__(): ", e_dataset.__len__())
#     #for i in range(e_dataset.__len__()):
#     #    seq_len,img_seq = e_dataset.__getitem__(i)
#         #print("Sequence length from __getitem__(): ", seq_len)
#         #print("Datatype of img_seq[-1]: ", type(img_seq[-1]))
#         #print("Shape of Tensor", img_seq[0][-1].size())

#     idx = [i for i in range(args.batch_size)]
#     print(idx)
#     seq_len,img_seq,pos,ori = e_dataset.__getitem__(idx)
#     # print("Sequence length from __getitem__(): ", seq_len)
#     # print("Datatype of img_seq[-1]: ", type(img_seq[-1]))
#     # print("Shape of Tensor", img_seq[-1].size())
#     # print("Shape of pos tensors", len(pos))
#     print(len(img_seq))

#     # img_seq = torch.stack(img_seq, dim=2)
#     # img_seq = img_seq.view(args.batch_size,seq_len,-1)

#     # print("Shape of img_seq after stacking and view", img_seq.size())


#     # print("Root dir: " + args.directory)
#     # print("list: " + args.train_files)
#     dataset = data(root_dir = args.directory, list_files = args.train_files, verbose_debug = False, sequence_length = args.max_length)
#     # print("Temp Dataset: ", dataset)
#     print("Temp Dataset __len__(): ", dataset.__len__())
#     #for i in range(e_dataset.__len__()):
#     #    seq_len,img_seq = e_dataset.__getitem__(i)
#         #print("Sequence length from __getitem__(): ", seq_len)
#         #print("Datatype of img_seq[-1]: ", type(img_seq[-1]))
#         #print("Shape of Tensor", img_seq[0][-1].size())


#     seq_len,img_seq,pos,ori = dataset.__getitem__(idx)
#     print(pos)
#     # print("Sequence length from __getitem__(): ", seq_len)
#     # print("Datatype of img_seq[-1]: ", type(img_seq[-1]))
#     # print("Shape of Tensor", img_seq[-1].size())
#     # print("Shape of pos tensors", len(pos))

#     tensor = torch.stack(img_seq, dim=1)
#     tensor = torch.cat([tensor[:,:-1], tensor[:,1:]],dim=2)
#     print("Shape of img_seq after make_pairs", tensor.size())
#     batch_size, time_steps, C, H, W, D = tensor.size()
#     tensor = tensor.view(batch_size*time_steps, C, H, W, D)
#     print("Shape of img_seq after stacking and view", tensor.size())

#     # print(img_seq)

#     # for i in range(len(img_seq)):
#     #     for j in range(len(img_seq[i])):
#     #         print(img_seq[i][j].size())

# #and seq_num + 1 -self.sequence_length <= 0

# #and seq_num+self.sequence_length is not len(imgs_sorted)