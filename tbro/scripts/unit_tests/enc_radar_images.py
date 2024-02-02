import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl

import numpy as np
from scipy.spatial.transform.rotation import Rotation as R

from scripts.models.deepro_enc import deepROEncoder
from scripts.utils.params import Parameters
from scripts.utils.losses import traj_and_odom_loss
from scripts.utils.params import Parameters

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class DeepROEncOnly(torch.nn.Module):
    def __init__(
        self, args: Parameters
    ):
        super().__init__()

        self.save_hyperparameters = {
            'sequence_length': args.max_length,
		    'batch_size': args.batch_size,
		    'epochs': args.epochs,
		    'radar_shape': args.radar_shape,
		    'hidden_size': args.hidden_size,
		    'learning_rate': args.learning_rate,
            'alphas': args.alphas,
            'betas': args.betas,
            'gammas': args.gammas
            }
        self.learning_rate = args.learning_rate

        # self.automatic_optimization = False
        # self.loss_func = traj_and_odom_loss(args.alphas, args.betas, args.gammas)

        self.mean = args.mean_enable
        self.learning_rate = args.learning_rate

        self.encoder = deepROEncoder(in_channels=2, max_pool = True) # Two radar images (power channel only)
        self._cnn_feature_vector_size = self.encoder.conv5_2[0].out_channels

        if args.pretrained_enc_path is not None:
            self.encoder.load_weights(args.pretrained_enc_path)
  

    def make_pairs(self, radar_images: List[torch.Tensor]) -> torch.Tensor:
        tensor = torch.stack(radar_images, dim=1)
        return torch.cat([tensor[:,:-1], tensor[:,1:]],dim=2)

    def forward_cnn(self, pairs: torch.Tensor) -> torch.Tensor:
        # batch_size, sequence_length, channels = 2, height = 64, width = 128, depth = 64
        batch_size, time_steps, C, H, W, D = pairs.size()
        c_in = pairs.view(batch_size*time_steps, C, H, W, D)
        encoded_radar_images = self.encoder(c_in)
        bt_size, C_out, _, _, _ = encoded_radar_images.size()
        if self.mean:
            cnn_out = torch.mean(encoded_radar_images, dim=[2,3,4]).view(batch_size, time_steps, C_out)
        else:
            cnn_out = encoded_radar_images
        return cnn_out
    
    def forward(self, radar_images: List[torch.Tensor]) -> torch.Tensor:
        encoded_images = self.forward_cnn(self.make_pairs(radar_images))
        return encoded_images

    def training_step(self,train_batch,batch_idx):
        seq_len, radar_data, positions, orientations = train_batch[0], train_batch[1], train_batch[2], train_batch[3]

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
    
    def validation_step(self,val_batch,batch_idx):
        seq_len, radar_data, positions, orientations = val_batch[0], val_batch[1], val_batch[2], val_batch[3]
                       
        y_hat = self.forward(radar_data)
        # loss = self.loss_func(y_hat,positions,orientations)
        # self.log('Validation_loss',loss,sync_dist=True)
        # return loss

class ModifiedDeepRODataset(Dataset): 
    def __init__(
        self, root_dir = None, list_files = None, 
        sequence_fname=None, load_sequence = False,
        sequence_length=None,
        verbose_debug=False
    ):
        super(ModifiedDeepRODataset,self).__init__()
        # Get root directory for the dataset and check formatting
        if root_dir is not None:
            self.root_dir = root_dir
            if self.root_dir[-1] != '/':
                self.root_dir = self.root_dir + '/'
        self.img_list = []

        self.verbose_debug = verbose_debug
        self.sequence_length = sequence_length

        if list_files is not None:
            self.load_img_list(list_files)
        
        if not load_sequence:
            self.img_sequences = self.create_sequences()
        else:
            self.load_sequences(sequence_fname)
    
    def load_img_list(self, filenames) -> List[str]:
        train_file_list = []
        dataset_file = open(self.root_dir + filenames)
        for file in dataset_file:
            train_file_list.append(file.strip() + '/img_list.txt')

        for filename in train_file_list:
            if self.verbose_debug:
                print(self.root_dir + filename)
            with open(self.root_dir + filename, 'r') as file:
                cur_list = []
                cur_list += file.readlines()
                cur_list = [s.strip() for s in cur_list]
                cur_list.sort(key=self.get_seq_num)
                self.img_list += cur_list

    def get_base_name(self, s) -> str:
        if self.verbose_debug:
            print("Getting base names")
            print(s[:s.rfind('_')])
        return s[:s.rfind('_')]

    def get_seq_num(self, s) -> int:
        if self.verbose_debug:
            print("Getting seq num")
            print(s[s.rfind('_')+1:])
        return int(s[s.rfind('_')+1:])

    def create_sequences(self) -> List[List[str]]:
        # separate image list into base names
        base_names = {}
        for img in self.img_list:
            name = self.get_base_name(img)
            if name in base_names.keys():
                base_names[name].append(img)
            else: 
                base_names[name] = [img]

        # generate all possible sequences within each base name
        sequences = []
        if self.sequence_length is not None:
            for name in base_names:
                imgs = base_names[name]
                imgs_sorted = sorted(imgs, key=lambda img: self.get_seq_num(img))

                for img in imgs_sorted: 
                    seq_num = self.get_seq_num(img)
                    if len(imgs_sorted) - seq_num >= self.sequence_length:
                        sequences.append([imgs_sorted[i] for i in range(seq_num, seq_num+self.sequence_length)])
                    if seq_num + 1 >= self.sequence_length:
                        sequences.append([imgs_sorted[i] for i in range(seq_num, seq_num-self.sequence_length,-1)])
        else:
            for name in base_names:
                imgs = base_names[name]
                img_sorted = sorted(imgs,key=lambda img: self.get_seq_num(img))
                sequences.append(img_sorted)
        return sequences


    def save_sequence(self,name):
        filename = self.rootdir + name+ '_sequence_list.txt'
        with open(filename, 'w') as file:
            for sequence in self.img_sequences:
                #TODO: Find a way to write sequences to individual files?
                file.write()
                file.write('\n')

    def load_sequence(self,name):
        filename = self.root_dir + name + '_sequence_list.txt'
        self.img_sequences = []
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.img_sequences.append(line.split(' '))

    def __len__(self) -> int:
        return len(self.img_sequences)

#    def __getitem__(self, indices):
#        return self.img_sequences[indices]


    def get_filenames(self,s):
        name = self.get_base_name(s[0])
        img_filenames = []
        pose_filenames = []
        for idx in range(len(s)):
            seq_idx = self.get_seq_num(s[idx])
            if self.verbose_debug:
                print("Sequence index=",seq_idx)
                print(os.path.join(self.root_dir, name + '_odom_data/images', str(seq_idx) + '.pt'))
            img_filenames.append(os.path.join(self.root_dir, name + '_odom_data/images', str(seq_idx) + '.pt'))
            pose_filenames.append(os.path.join(self.root_dir, name + '_odom_data/poses', str(seq_idx) + '.pt'))


        return {'imgs':img_filenames, 
                'poses':pose_filenames,
                }

    def __getitem__(self,indices) -> Tuple[int,List[torch.tensor],List[torch.tensor],List[torch.tensor],List[str]]:
        # Format indices to list
        if torch.is_tensor(indices):
            indices = indices.to_list()
        if isinstance(indices, int):
            indices = [indices]
        
        for idx in indices:
            # dictionary of filenames stored under keys 'imgs' and 'poses'
            filenames = self.get_filenames(self.img_sequences[idx])

            # Generate Image Sequences
            img_seq = []
            seq_len = len(filenames['imgs'])
            for filename in filenames['imgs']:
                if self.verbose_debug: 
                    print(filename)
                img_seq.append(torch.load(filename))
            

            # Generate GT Sequences
            positions = []
            orientations = []
            for i in range(len(filenames['poses'])-1):
                T_wsk = torch.load(filenames['poses'][i])
                T_wsk1 = torch.load(filenames['poses'][i+1])
                T_sksk1 = torch.matmul(T_wsk.inverse(),T_wsk1)
                position = T_sksk1[:3,3]
                r = R.from_matrix(T_sksk1[:3,:3].numpy())
                orientation = r.as_euler('xyz')
                orientation_tensor = torch.from_numpy(np.ascontiguousarray(orientation)).float()
                positions.append(position)
                orientations.append(orientation_tensor)

            combined_filenames = []
            for filename in filenames['imgs']:
                combined_filenames.append(filename)                 
        
        return (seq_len,img_seq,positions,orientations,combined_filenames)

def get_base_name(s) -> str:
    # if self.verbose_debug:
    #     print("Getting base names")
    #     print(s[:s.rfind('_')])
    return s[:s.rfind('_')]

def get_elem_num(s) -> int:
    #print(s[s.rfind('.pt')])
    string_temp = s.split('/')
    string_temp = string_temp[-1]
    string_temp = string_temp.split('.')
    string_temp = string_temp[0]
    #print(string_temp)

    # if self.verbose_debug:
        # print("Getting seq num")
        # print(s[s.rfind('_')+1:])
    return int(string_temp)

if __name__== '__main__':
    args = Parameters()
    
    dataset = ModifiedDeepRODataset(root_dir = args.directory, list_files = args.train_files, verbose_debug = False, sequence_length = args.max_length)
    print('Loaded ' + str(len(dataset)) + ' training sequences.')
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=False)

    device = torch.device("cuda")
    model = DeepROEncOnly(args)
    model.to(device)
    # wandb_logger = WandbLogger(project="TBRO")
    # trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu',devices=2,strategy=DDPStrategy(find_unused_parameters=False),max_epochs=args.epochs)
    # trainer = pl.Trainer(logger=wandb_logger,accelerator='gpu',devices=1,max_epochs=args.epochs)
    # trainer.test(model,data_module)

    for i, data in enumerate(loader):
        seq_len, radar_data, positions, orientations, filenames = data[0], data[1], data[2], data[3], data[4]
        output_tensor_list = []
        for k in range(seq_len):
            radar_data[k] = radar_data[k].to(device)
        with torch.no_grad():
            enc_image = model.forward(radar_data)
            # enc_image = enc_image.squeeze()
        # output_tensor_list.append(enc_image)

        # odom_positions = torch.stack(positions, dim=1)
        # odom_orientations = torch.stack(orientations, dim=1)
        # gt_odom_positions = odom_positions.squeeze()
        # gt_odom_orientations = odom_orientations.squeeze()

        # output_tensor_list.append(gt_odom_positions)
        # output_tensor_list.append(gt_odom_orientations)

        if(get_elem_num(filenames[0][0])<get_elem_num(filenames[1][0])):
            # print(get_base_name(filenames[0][0]) + '_data/encoded_images/forward_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
            path = get_base_name(filenames[0][0]) + '_data/encoded_images/forward_seq/'
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(enc_image, path + get_elem_num(filenames[0][0]).__str__() + '.pt')
            # pos_path = path + '/pos/'
            # ori_path = path + '/ori/'
            # if not os.path.exists(pos_path):
            #     os.makedirs(pos_path)
            # if not os.path.exists(ori_path):
            #     os.makedirs(ori_path)
            # torch.save(positions, pos_path + get_elem_num(filenames[0][0]).__str__() + '.pt')
            # torch.save(orientations, ori_path + get_elem_num(filenames[0][0]).__str__() + '.pt')

        elif(get_elem_num(filenames[0][0])>get_elem_num(filenames[1][0])):
            # print(get_base_name(filenames[0][0]) + '_data/encoded_images/reverse_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
            path = get_base_name(filenames[0][0]) + '_data/encoded_images/reverse_seq/'
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(enc_image,get_base_name(filenames[0][0]) + '_data/encoded_images/reverse_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
        else:
            print("Error occured on" + filenames)

        if i % 100 == 0:
            print('Processed ' + i.__str__() + ' of ' + str(len(dataset)) + 'samples')



        # if i < 2:
        #     print(filenames)
            # print(get_base_name(filenames[0][0]))
            # print(radar_data[0].size())
            # enc_image = model.forward(radar_data)
            # print(enc_image.size())
            # enc_image = enc_image.squeeze()
            # print(enc_image.size())
            # output_tensor_list.append(enc_image)
            

            # odom_positions = torch.stack(positions, dim=1)
            # odom_orientations = torch.stack(orientations, dim=1)
            # gt_odom_positions = odom_positions.squeeze()
            # gt_odom_orientations = odom_orientations.squeeze()

            # output_tensor_list.append(gt_odom_positions)
            # output_tensor_list.append(gt_odom_orientations)

            # print(len(output_tensor_list))
            # print(get_base_name(filenames[0][0]) + '_data/encoded_images/')
            # print(get_elem_num(filenames[0][0]))
            # print(get_elem_num(filenames[1][0]))
            # if(get_elem_num(filenames[0][0])<get_elem_num(filenames[1][0])):
            #     print(get_base_name(filenames[0][0]) + '_data/encoded_images/forward_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
            #     path = get_base_name(filenames[0][0]) + '_data/encoded_images/forward_seq/'
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     torch.save(output_tensor_list,get_base_name(filenames[0][0]) + '_data/encoded_images/forward_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
            # elif(get_elem_num(filenames[0][0])>get_elem_num(filenames[1][0])):
            #     print(get_base_name(filenames[0][0]) + '_data/encoded_images/reverse_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
            #     path = get_base_name(filenames[0][0]) + '_data/encoded_images/reverse_seq/'
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     torch.save(output_tensor_list,get_base_name(filenames[0][0]) + '_data/encoded_images/reverse_seq/' + get_elem_num(filenames[0][0]).__str__() + '.pt')
            # else:
            #     print("Error")
            

            


