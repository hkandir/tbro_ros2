"""
Povides custom pytorch data class for TBRO ROS node
"""

import logging
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

logging.basicConfig(level=logging.INFO)


class ThreeDSample(nn.Module):
    def __init__(self, method="trilinear"):
        super(ThreeDSample, self).__init__()
        self.methods = ["nearest", "trilinear"]
        self.method = method
        if self.method not in self.methods:
            raise RuntimeError("invalid interpolation method")

        self.pad = nn.ReplicationPad3d(padding=[0, 1, 0, 1, 0, 1])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p001 = torch.tensor([[[[0], [0], [1]]]], device=self.device)
        self.p011 = torch.tensor([[[[0], [1], [1]]]], device=self.device)
        self.p010 = torch.tensor([[[[0], [1], [0]]]], device=self.device)
        self.p100 = torch.tensor([[[[1], [0], [0]]]], device=self.device)
        self.p110 = torch.tensor([[[[1], [1], [0]]]], device=self.device)
        self.p101 = torch.tensor([[[[1], [0], [1]]]], device=self.device)

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
        # values = self.pad(values)

        # add zero padding
        values = F.pad(values, (2, 2, 2, 2, 2, 2), "constant", 0)

        # shift points by 2 to account for padding
        points = points + 2
        # 1 2 3 4
        # 0 0 1 1 2 3 4
        # clamp points to just outside the original range of the vaules tensor
        # so points outside that range will interpolate to zero
        points[:, :, 0, :] = points[:, :, 0, :].clamp(0, depth + 2)
        points[:, :, 1, :] = points[:, :, 1, :].clamp(0, width + 2)
        points[:, :, 2, :] = points[:, :, 2, :].clamp(0, height + 2)

        b = torch.tensor(range(batch_size)).unsqueeze(-1).repeat(1, num_points)

        if self.method is self.methods[0]:

            p = torch.round(points).long()
            return values[b, :, p[:, :, 0, 0], p[:, :, 1, 0], p[:, :, 2, 0]]

        elif self.method is self.methods[1]:

            # get surrounding pixel locations (size batch_size x 1 x 3 x 1)
            dp = points - points.floor()  # batch_size x num_points x 3 x 1
            c000 = points.floor().long()
            c001 = c000 + self.p001.view(1, 1, 3, 1)
            c011 = c000 + self.p011.view(1, 1, 3, 1)
            c010 = c000 + self.p010.view(1, 1, 3, 1)
            c100 = c000 + self.p100.view(1, 1, 3, 1)
            c110 = c000 + self.p110.view(1, 1, 3, 1)
            c101 = c000 + self.p101.view(1, 1, 3, 1)
            c111 = points.ceil().long()

            # get surrounding voxel values (size batch_size x num_points x channels)
            d000 = values[b, :, c000[:, :, 0, 0], c000[:, :, 1, 0], c000[:, :, 2, 0]]
            d001 = values[b, :, c001[:, :, 0, 0], c001[:, :, 1, 0], c001[:, :, 2, 0]]
            d011 = values[b, :, c011[:, :, 0, 0], c011[:, :, 1, 0], c011[:, :, 2, 0]]
            d010 = values[b, :, c010[:, :, 0, 0], c010[:, :, 1, 0], c010[:, :, 2, 0]]
            d100 = values[b, :, c100[:, :, 0, 0], c100[:, :, 1, 0], c100[:, :, 2, 0]]
            d110 = values[b, :, c110[:, :, 0, 0], c110[:, :, 1, 0], c110[:, :, 2, 0]]
            d101 = values[b, :, c101[:, :, 0, 0], c101[:, :, 1, 0], c101[:, :, 2, 0]]
            d111 = values[b, :, c111[:, :, 0, 0], c111[:, :, 1, 0], c111[:, :, 2, 0]]

            # interpolate descriptor values to point locations (size batch_size x num_points x channels)
            d00 = d000 * (1.0 - dp[:, :, 0, :]) + d100 * dp[:, :, 0, :]
            d01 = d001 * (1.0 - dp[:, :, 0, :]) + d101 * dp[:, :, 0, :]
            d10 = d010 * (1.0 - dp[:, :, 0, :]) + d110 * dp[:, :, 0, :]
            d11 = d011 * (1.0 - dp[:, :, 0, :]) + d111 * dp[:, :, 0, :]
            d0 = d00 * (1.0 - dp[:, :, 1, :]) + d10 * dp[:, :, 1, :]
            d1 = d01 * (1.0 - dp[:, :, 1, :]) + d11 * dp[:, :, 1, :]

            return d0 * (1.0 - dp[:, :, 2, :]) + d1 * dp[:, :, 2, :]


# globals
# TODO: make this not global
im_coords = None
im_points = None
verbose_debug = True
sampler = ThreeDSample()


def polar_to_cartesian(r, az, el):
    x = r * math.cos(el) * math.cos(az)
    y = r * math.cos(el) * math.sin(az)
    z = r * math.sin(el)
    return (x, y, z)


# returns the interpolated index of the value in the input array
# where the input value would be
def get_array_idx(arr, val):

    lower_idx = min(range(len(arr)), key=lambda i: abs(arr[i] - val))
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

    range_bin_width = bin_widths[0]  # scalar
    azimuth_bins = bin_widths[1]  # array
    elevation_bins = bin_widths[2]  # array

    x_dim = dim[0]
    y_dim = dim[1]
    z_dim = dim[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if im_points is None:
        im_points = torch.zeros(x_dim, y_dim, z_dim, 3, device=device)
        for i in range(x_dim):
            x = vox_width * float(i)
            for j in range(y_dim):
                y = vox_width * (float(j) - (float(y_dim) - 1.0) / 2.0)
                for k in range(z_dim):
                    z = vox_width * (float(k) - (float(z_dim) - 1.0) / 2.0)
                    r = math.sqrt(x**2 + y**2 + z**2)
                    phi = math.atan2(z, math.sqrt(x**2 + y**2))
                    theta = math.atan2(y, x)

                    r_bin = r / range_bin_width
                    theta_bin = get_array_idx(azimuth_bins, theta)
                    phi_bin = get_array_idx(elevation_bins, phi)

                    im_points[i, j, k, :] = torch.tensor([r_bin, theta_bin, phi_bin])

        im_points = im_points.view(1, -1, 3, 1).to(device)
        if verbose_debug:
            print("Shape of cartesian tensor:", im_points.shape)

    polar_img = torch.from_numpy(polar_img[0:1, :, :, :])
    polar_img = polar_img.to(device)

    # need to verify sampler function with multi-channel images
    cartesian_arr = sampler(
        im_points,
        polar_img.view(
            1,
            polar_img.shape[0],
            polar_img.shape[1],
            polar_img.shape[2],
            polar_img.shape[3],
        ),
    )
    if verbose_debug:
        print("Shape of carteisan array:", cartesian_arr.shape)
    cartesian_img = cartesian_arr.transpose(1, 2).view(1, x_dim, y_dim, z_dim)

    return cartesian_img.float().cpu()


def process_img(msg):
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
                angle_idx = az_idx + width * el_idx
                src_idx = 2 * (range_idx + depth * angle_idx)
                img[0, range_idx, az_idx, el_idx] = arr[src_idx]
                img[1, range_idx, az_idx, el_idx] = arr[src_idx + 1]

    if im_coords is None:
        im_coords = np.zeros((3, depth, width, height))
        for range_idx in range(depth):
            r = range_idx * range_bin_width
            for az_idx in range(width):
                az = az_bins[az_idx]
                for el_idx in range(height):
                    el = el_bins[el_idx]
                    im_coords[:, range_idx, az_idx, el_idx] = polar_to_cartesian(
                        r, az, el
                    )
    img = np.concatenate((img, im_coords), axis=0)

    bin_widths = (msg.range_bin_width, msg.azimuth_bins, msg.elevation_bins)
    cartesian_img = resample_to_cartesian(img, (64, 128, 64), 0.12, bin_widths)

    """
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
  """

    # img = torch.from_numpy(img).float()

    # return (msg.header.stamp.sec, cartesian_img)
    return cartesian_img


class MimoDataset(Dataset):
    def __init__(self):
        super(MimoDataset, self).__init__()
        self.image_list = []

    def load_img(self, new_mimo_msg):
        logging.debug(
            "mimo:loading new_mimo_msg:image:size: {}".format(len(new_mimo_msg.image))
        )
        if len(self.image_list) > 1:
            self.image_list.pop(0)
        self.image_list.append(process_img(new_mimo_msg))

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index):
        # image = torch.FloatTensor(self.image_list[index])
        image = self.image_list[index]
        lable = ...

        return image, lable
