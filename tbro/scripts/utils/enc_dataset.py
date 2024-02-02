import os
import shutil

from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scripts.utils.params import Parameters

class DeepRODataset(Dataset): 
    def __init__(
        self, 
        root_dir = None, 
        list_files = None, 
        sequence_fname = None, 
        load_sequence = False,
        sequence_length = None,
        verbose_debug = False,
        forward_sequence_only = False
    ):
        super(DeepRODataset,self).__init__()
        # Get root directory for the dataset and check formatting
        if root_dir is not None:
            self.root_dir = root_dir
            if self.root_dir[-1] != '/':
                self.root_dir = self.root_dir + '/'
        self.img_list = []

        self.verbose_debug = verbose_debug
        self.sequence_length = sequence_length
        self.forward_sequence_only = forward_sequence_only

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
                    if (len(imgs_sorted)-1) - seq_num >= self.sequence_length:
                        sequences.append([imgs_sorted[i] for i in range(seq_num, seq_num+self.sequence_length)])
                    if seq_num + 1 >= self.sequence_length and not self.forward_sequence_only:
                        sequences.append([imgs_sorted[i] for i in range(seq_num, seq_num-self.sequence_length,-1)])
        else:
            for name in base_names:
                imgs = base_names[name]
                img_sorted = sorted(imgs,key=lambda img: self.get_seq_num(img))
                sequences.append(img_sorted)
                
        if self.verbose_debug:
            print(sequences)
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
        # This is the file to edit:
        # Forward sequences (i.e. counting up) need to load from forward
        # Backward sequences  (i.e. counting down) need to load from reverse folders
        # Can return just 'imgs' as previous (?)
        name = self.get_base_name(s[0])
        img_filenames = []
        pose_filenames = []
        # img_reverse_filenames = []
        seq_idx_test = []
        for idx in range(len(s)):
            seq_idx_temp = self.get_seq_num(s[idx])
            seq_idx_test.append(seq_idx_temp)
        
        forward_bool = True
        if len(seq_idx_test) > 1:
            if seq_idx_test[0] < seq_idx_test[1]:
                forward_bool = True
            elif seq_idx_test[0] > seq_idx_test[1]:
                forward_bool = False
            else:
                print("Sequence Error: idx and idx+1 are equal")


        for idx in range(len(s)):
            seq_idx = self.get_seq_num(s[idx])
            pose_filenames.append(os.path.join(self.root_dir, name + '_odom_data/poses', str(seq_idx) + '.pt'))
            if forward_bool:
                if self.verbose_debug:
                    print("Sequence index=",seq_idx)
                    print(os.path.join(self.root_dir, name + '_odom_data/encoded_images/forward_seq/', str(seq_idx) + '.pt'))
                img_filenames.append(os.path.join(self.root_dir, name + '_odom_data/encoded_images/forward_seq/', str(seq_idx) + '.pt'))
                if idx == len(s)-1:
                    pose_filenames.append(os.path.join(self.root_dir, name + '_odom_data/poses', str(seq_idx+1) + '.pt'))
            else:
                if self.verbose_debug:
                    print("Sequence index=",seq_idx)
                    print(os.path.join(self.root_dir, name + '_odom_data/encoded_images/reverse_seq/', str(seq_idx) + '.pt'))
                img_filenames.append(os.path.join(self.root_dir, name + '_odom_data/encoded_images/reverse_seq/', str(seq_idx) + '.pt'))
                if idx == len(s)-1:
                    pose_filenames.append(os.path.join(self.root_dir, name + '_odom_data/poses', str(seq_idx-1) + '.pt'))
          
        return {'imgs':img_filenames,
                'poses':pose_filenames,
                }

    def __getitem__(self,indices) -> Tuple[int,List[torch.tensor],List[torch.tensor],List[torch.tensor]]:
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
            # print(filenames)
            for filename in filenames['imgs']:
                if self.verbose_debug: 
                    print(filename)
                tensor = torch.load(filename,map_location='cpu')
                # tensor = torch.unsqueeze(tensor,dim=0)
                tensor = torch.squeeze(tensor)
                img_seq.append(tensor)
            

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
        
        return (seq_len,img_seq,positions,orientations)
        # return (seq_len,img_seq)

class DeepRODataModule(pl.LightningDataModule):
    def __init__(self,args: Parameters):
        self.args = args
        self.prepare_data_per_node=False
        self.save_hyperparameters()
        # self.prepare_data()
        # self.setup()

    def setup(self, stage = None):
        print(self.args.directory)
        self.train_dataset = DeepRODataset(root_dir = self.args.directory, list_files = self.args.train_files, verbose_debug = False, sequence_length = self.args.max_length, forward_sequence_only = self.args.forward_seq_only)
        print('Loaded ' + str(len(self.train_dataset)) + ' training sequences.')
        self.test_dataset = DeepRODataset(root_dir = self.args.directory, list_files = self.args.val_files, verbose_debug = False, sequence_length = self.args.max_length, forward_sequence_only = self.args.forward_seq_only)
        print('Loaded ' + str(len(self.test_dataset)) + ' testing sequences.')


    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,batch_size=self.args.batch_size,shuffle=True,num_workers=4,drop_last=True)
        return train_loader

    def val_dataloader(self):
        test_loader = DataLoader(self.test_dataset,batch_size=self.args.batch_size,shuffle=False,num_workers=4,drop_last=True)
        return test_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,batch_size=self.args.batch_size,shuffle=False,num_workers=4,drop_last=True)
        return test_loader

        






