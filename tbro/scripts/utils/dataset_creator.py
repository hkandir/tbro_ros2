import sys
import argparse
import os

import math
import numpy as np

import torch
import torch.nn  as nn
import torch.nn.functional as F

import scipy.spatial.transform as tf 
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d

import rosbag

from nav_msgs.msg import Odometry
from dca1000_device_msgs.msg import MimoMsg

#from deep_ro_dataset import DeepRODataset 
from utils.params import Parameters

class ThreeDSample(nn.Module):
  def __init__(self, method='trilinear'):
    super(ThreeDSample, self).__init__()
    self.methods = ['nearest', 'trilinear']
    self.method = method
    if self.method not in self.methods:
      raise RuntimeError('invalid interpolation method')

    self.pad = nn.ReplicationPad3d(padding=[0,1,0,1,0,1])

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.p001 = torch.tensor([[[[0],[0],[1]]]],device=self.device)
    self.p011 = torch.tensor([[[[0],[1],[1]]]],device=self.device)
    self.p010 = torch.tensor([[[[0],[1],[0]]]],device=self.device)
    self.p100 = torch.tensor([[[[1],[0],[0]]]],device=self.device)
    self.p110 = torch.tensor([[[[1],[1],[0]]]],device=self.device)
    self.p101 = torch.tensor([[[[1],[0],[1]]]],device=self.device)

  # accepts a tensor of size (batch_size,num_samples,3,1) describing the locations 
  # at which samples should be taken
  # and a tensor of size (batch_size,channels,depth,width,height) describing an n-channel 3d image
  # returns values for each sample point using the specified method
  def forward(self, points, values):
    batch_size = points.shape[0]
    num_points = points.shape[1]
    depth = values.shape[2]
    width = values.shape[3]
    height = values.shape[4]

    # replication pad so points with index == dim will evaluate correctly
    #values = self.pad(values)

    # add zero padding
    values = F.pad(values,(2,2,2,2,2,2),"constant",0)

    # shift points by 2 to account for padding
    points = points + 2
    # 1 2 3 4
    # 0 0 1 1 2 3 4
    # clamp points to just outside the original range of the vaules tensor
    # so points outside that range will interpolate to zero
    points[:,:,0,:] = points[:,:,0,:].clamp(0,depth+2)
    points[:,:,1,:] = points[:,:,1,:].clamp(0,width+2)
    points[:,:,2,:] = points[:,:,2,:].clamp(0,height+2)
    
    b = torch.tensor(range(batch_size)).unsqueeze(-1).repeat(1,num_points)

    if self.method is self.methods[0]:

      p = torch.round(points).long()
      return values[b,:,p[:,:,0,0],p[:,:,1,0],p[:,:,2,0]]

    elif self.method is self.methods[1]:

      # get surrounding pixel locations (size batch_size x 1 x 3 x 1)
      dp = points - points.floor() # batch_size x num_points x 3 x 1
      c000 = points.floor().long()
      c001 = c000 + self.p001.view(1,1,3,1)
      c011 = c000 + self.p011.view(1,1,3,1)
      c010 = c000 + self.p010.view(1,1,3,1)
      c100 = c000 + self.p100.view(1,1,3,1)
      c110 = c000 + self.p110.view(1,1,3,1)
      c101 = c000 + self.p101.view(1,1,3,1)
      c111 = points.ceil().long()


      # get surrounding voxel values (size batch_size x num_points x channels)
      d000 = values[b,:,c000[:,:,0,0],c000[:,:,1,0],c000[:,:,2,0]]
      d001 = values[b,:,c001[:,:,0,0],c001[:,:,1,0],c001[:,:,2,0]]
      d011 = values[b,:,c011[:,:,0,0],c011[:,:,1,0],c011[:,:,2,0]]
      d010 = values[b,:,c010[:,:,0,0],c010[:,:,1,0],c010[:,:,2,0]]
      d100 = values[b,:,c100[:,:,0,0],c100[:,:,1,0],c100[:,:,2,0]]
      d110 = values[b,:,c110[:,:,0,0],c110[:,:,1,0],c110[:,:,2,0]]
      d101 = values[b,:,c101[:,:,0,0],c101[:,:,1,0],c101[:,:,2,0]]
      d111 = values[b,:,c111[:,:,0,0],c111[:,:,1,0],c111[:,:,2,0]]

      # interpolate descriptor values to point locations (size batch_size x num_points x channels)
      d00 = d000 * (1.0 - dp[:,:,0,:]) + d100 * dp[:,:,0,:]
      d01 = d001 * (1.0 - dp[:,:,0,:]) + d101 * dp[:,:,0,:]
      d10 = d010 * (1.0 - dp[:,:,0,:]) + d110 * dp[:,:,0,:]
      d11 = d011 * (1.0 - dp[:,:,0,:]) + d111 * dp[:,:,0,:]
      d0 = d00 * (1.0 - dp[:,:,1,:]) + d10 * dp[:,:,1,:]
      d1 = d01 * (1.0 - dp[:,:,1,:]) + d11 * dp[:,:,1,:]

      return d0 * (1.0 - dp[:,:,2,:]) + d1 * dp[:,:,2,:]

# read raw messages from a rosbag
def read_bag(filename, radar_topic, odom_topic):
  bag = rosbag.Bag(filename)
  msgs = {'radar':[], 'odom':[]}

  print('reading ' + filename)

  print('getting odom messages')
  bag_msgs = bag.read_messages(topics=[odom_topic])
  msgs['odom'] = [process_odom(msg,t) for (topic,msg,t) in bag_msgs]
  msgs['odom'] = [odom_msg for odom_msg in msgs['odom'] if odom_msg is not None]

  print('getting radar messages')
  bag_msgs = bag.read_messages(topics=[radar_topic])
  msgs['radar'] = [process_img(msg,t) for (topic,msg,t) in bag_msgs]

  max_intensity = 0
  min_intensity = 1e10
  for radar_im in msgs['radar']:
    if radar_im[1][0,:,:,:].max() > max_intensity:
      max_intensity = radar_im[1][0,:,:,:].max()
    if radar_im[1][0,:,:,:].min() < min_intensity:
      min_intensity = radar_im[1][0,:,:,:].min()

  for radar_im in msgs['radar']:
    radar_im[1][0,:,:,:] -= min_intensity
    radar_im[1][0,:,:,:] /= (max_intensity - min_intensity)

  msgs['odom'].sort()
  msgs['radar'].sort()

  return msgs

def process_odom(msg, t):
  t= np.array([msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               msg.pose.pose.position.z])

  q_arr = np.array([msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w])

  if np.any(np.isnan(t)) or np.any(np.isnan(q_arr)): return None

  q = tf.Rotation.from_quat(q_arr)

  return ((msg.header.stamp.to_sec(), t, q))


# returns the interpolated index of the value in the input array 
# where the input value would be
def get_array_idx(arr, val):

  lower_idx = min(range(len(arr)), key = lambda i: abs(arr[i]-val))
  if arr[lower_idx] > val:
    lower_idx -= 1

  if lower_idx < 0:
    return float(lower_idx)
  elif lower_idx == len(arr) - 1:
    if val > arr[lower_idx]:
      return float(lower_idx + 1)
    else:
      return float(lower_idx)

  upper_idx = lower_idx + 1
  lower_val = arr[lower_idx]
  upper_val = arr[upper_idx]

  c = (val - lower_val) / (upper_val - lower_val)
  idx = lower_idx * (1.0 - c) + upper_idx * c

  return idx


# transforms polar image with min range, azimuth, and elevation at (0,0,0)
# to cartesian image with min x, y and z at (0,0,0)
def resample_to_cartesian(polar_img, dim, vox_width, bin_widths):
  global sampler
  global im_points
  global verbose_debug

  range_bin_width = bin_widths[0] # scalar
  azimuth_bins = bin_widths[1]    # array
  elevation_bins = bin_widths[2]  # array

  x_dim = dim[0]
  y_dim = dim[1]
  z_dim = dim[2]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if im_points is None:
    im_points = torch.zeros(x_dim,y_dim,z_dim,3,device=device)
    for i in range(x_dim):
      x = vox_width * float(i)
      for j in range(y_dim):
        y = vox_width * (float(j) - (float(y_dim)-1.) / 2.0)
        for k in range(z_dim):
          z = vox_width * (float(k) - (float(z_dim)-1.) / 2.0)
          r = math.sqrt(x**2 + y**2 + z**2)
          phi = math.atan2(z,math.sqrt(x**2 + y**2))
          theta = math.atan2(y,x)

          r_bin = r / range_bin_width
          theta_bin = get_array_idx(azimuth_bins, theta)
          phi_bin = get_array_idx(elevation_bins, phi)

          im_points[i,j,k,:] = torch.tensor([r_bin,theta_bin,phi_bin])

    im_points = im_points.view(1,-1,3,1).to(device)
    if verbose_debug:
      print("Shape of cartesian tensor:", im_points.shape)

  polar_img = torch.from_numpy(polar_img[0:1,:,:,:])
  polar_img = polar_img.to(device)

  # need to verify sampler function with multi-channel images
  cartesian_arr = sampler(im_points, 
                          polar_img.view(1,
                                         polar_img.shape[0],
                                         polar_img.shape[1],
                                         polar_img.shape[2],
                                         polar_img.shape[3]))
  if verbose_debug:
    print("Shape of carteisan array:", cartesian_arr.shape)
  cartesian_img = cartesian_arr.transpose(1,2).view(1,x_dim,y_dim,z_dim)

  return cartesian_img.float().cpu()
  

#def downsample_pointcloud(in_pcl, vox_size):
#  _, idx = np.unique((in_pcl[2:,:] / vox_size).round(),return_index=True,axis=1)
#  out_pcl = in_pcl[:,idx]
#  return out_pcl


def polar_to_cartesian(r, az, el):
  x = r * math.cos(el) * math.cos(az)
  y = r * math.cos(el) * math.sin(az)
  z = r * math.sin(el)
  return (x, y, z)


def process_img(msg,t):
  global im_coords
  width = msg.width
  height = msg.height
  depth = msg.depth
  az_bins = msg.azimuth_bins
  el_bins = msg.elevation_bins
  range_bin_width = msg.range_bin_width
  arr = msg.image
  
  img = np.zeros((2, depth, width, height))
  for range_idx in range(depth):
    for az_idx in range(width):
      for el_idx in range(height):
        angle_idx = az_idx + width * el_idx;
        src_idx = 2 * (range_idx + depth * angle_idx)
        img[0,range_idx,az_idx,el_idx] = arr[src_idx]
        img[1,range_idx,az_idx,el_idx] = arr[src_idx+1]

  if im_coords is None:
    im_coords = np.zeros((3, depth, width, height))
    for range_idx in range(depth):
      r = range_idx * range_bin_width
      for az_idx in range(width):
        az = az_bins[az_idx]
        for el_idx in range(height):
          el = el_bins[el_idx]
          im_coords[:, range_idx, az_idx, el_idx] = polar_to_cartesian(r, az, el)
  img = np.concatenate((img, im_coords), axis=0)

  bin_widths = (msg.range_bin_width, msg.azimuth_bins, msg.elevation_bins)
  cartesian_img = resample_to_cartesian(img, (64,128,64), 0.12, bin_widths)
  
  '''
  img = np.zeros((5, depth, width))
  for range_idx in range(depth):
    r = range_idx * range_bin_width
    for az_idx in range(width):
      az = az_bins[az_idx]

      max_el_i = 0
      max_el_d = 0
      max_el_bin = 0
      for el_idx in range(height):
        angle_idx = az_idx + width * el_idx
        src_idx = 2 * (range_idx + depth * angle_idx)
        if arr[src_idx] > max_el_i:
          max_el_i = arr[src_idx]
          max_el_d = arr[src_idx+1]
          max_el_bin = el_idx

      el = el_bins[max_el_bin]
      img[2:, range_idx, az_idx] = polar_to_cartesian(r, az, el)
      img[0, range_idx, az_idx] = max_el_i
      img[1, range_idx, az_idx] = max_el_d
  '''

  #img = torch.from_numpy(img).float()

  return (msg.header.stamp.to_sec(), cartesian_img)


# interpolate odometry messages to align temporally with radar messages
# also transform odom from base frame to radar frame
def temporal_align(msgs, T_br):

  synced_msgs = [] 

  # ensure there are odom messages before the first radar message
  # and after the last radar message
  while len(msgs['radar']) > 0 and msgs['odom'][0][0] > msgs['radar'][0][0]:
    msgs['radar'].pop(0)
  while len(msgs['radar']) > 0 and msgs['odom'][-1][0] < msgs['radar'][-1][0]:
    msgs['radar'].pop(-1)

  odom_idx = 0
  radar_idx = 0

  while radar_idx < len(msgs['radar']):

    radar_stamp = msgs['radar'][radar_idx][0]

    # find odom messages that bracket the current radar message
    # prior check should prevent running off the end of the odom msgs
    while not (msgs['odom'][odom_idx][0] <= radar_stamp 
      and msgs['odom'][odom_idx+1][0] >= radar_stamp):
      odom_idx += 1

    before_stamp = msgs['odom'][odom_idx][0]
    after_stamp = msgs['odom'][odom_idx+1][0]
    before_t = msgs['odom'][odom_idx][1]
    after_t = msgs['odom'][odom_idx+1][1]
    before_q = msgs['odom'][odom_idx][2]
    after_q = msgs['odom'][odom_idx+1][2]

    c = (radar_stamp - before_stamp) / (after_stamp - before_stamp)

    t = before_t * (1.0 - c) + after_t * c
    slerp = tf.Slerp([0.0,1.0],tf.Rotation([before_q.as_quat(),after_q.as_quat()]))
    q = slerp([c])[0]
    R = q.as_matrix()

    # get 4x4 matrix for base-to-world transformation
    T_wb = np.eye(4)
    T_wb[:3,:3] = R
    T_wb[:3,3] = t

    # get radar-to-world transformation
    T_wr = np.dot(T_wb, T_br)

    synced_msgs.append({'timestamp':radar_stamp,
                        'img':msgs['radar'][radar_idx][1],
                        'pose':torch.from_numpy(T_wr).float()})

    radar_idx += 1

  return synced_msgs


def read_tf_file(filename):

  T_br = np.eye(4)

  if not os.path.exists(filename):
    print('File ' + filename + ' not found')
    return T_br

  with open(filename, mode='r') as file:
    lines = file.readlines()

  t = [float(s) for s in lines[0].split()]
  q = [float(s) for s in lines[1].split()]

  rot = tf.Rotation.from_quat(q)
  R = rot.as_matrix()
  
  T_br[:3,:3] = R
  T_br[:3,3] = t

  return T_br


if __name__ == "__main__":

  im_points = None
  im_coords = None
  fixed_points = None
  sampler = ThreeDSample()
  verbose_debug = True

  args = Parameters()

  if args.directory[-1] != '/': args.directory = args.directory + '/'
  
  T_br = read_tf_file(args.directory + args.base_to_radar)
  bag_list = open(args.directory + args.bags)
  for bag_name in bag_list:
    print('getting messages')
    msgs = read_bag(args.directory + bag_name.strip(),
                    args.radar_topic,
                    args.odom_topic)
    print('syncing messages')
    msgs_aligned = temporal_align(msgs, T_br)

    print('creating data tuples')
    dataset = DeepRODataset(root_dir=args.directory)
    for i in range(len(msgs_aligned)):
      # get pose relative to initial pose
      dataset.add_item(msgs_aligned[i]['img'],
                       msgs_aligned[i]['pose'],
                       bag_name.strip(),
                       i)
    dataset.save_img_list(bag_name.strip())

    del msgs
    del msgs_aligned
    del dataset

  bag_list.close()
