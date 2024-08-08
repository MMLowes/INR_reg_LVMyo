import numpy as np
import os
import torch
import SimpleITK as sitk
import edt
  

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output

def fast_trilinear_interpolation_4D(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0].T * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0].T * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0].T * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1].T * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1].T * x * (1 - y) * z
        + input_array[x0, y1, z1].T * (1 - x) * y * z
        + input_array[x1, y1, z0].T * x * y * (1 - z)
        + input_array[x1, y1, z1].T * x * y * z
    )
    return output.T

def fast_nearest_neighbor_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x_indices = torch.round(x_indices).to(torch.long)
    y_indices = torch.round(y_indices).to(torch.long)
    z_indices = torch.round(z_indices).to(torch.long)

    x_indices = torch.clamp(x_indices, 0, input_array.shape[0] - 1)
    y_indices = torch.clamp(y_indices, 0, input_array.shape[1] - 1)
    z_indices = torch.clamp(z_indices, 0, input_array.shape[2] - 1)

    output = input_array[x_indices, y_indices, z_indices]
    return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor



######################################
import re
import pandas as pd
from scipy.ndimage import binary_dilation
from utils import utils

def get_path_and_root(use_MI_data=False):
    user = os.getlogin()

    # local user
    if user == 'lowes':
        RH = False
        root = r"C:\Users\lowes\OneDrive\Skrivebord\DTU\phd\DTU-CFA-PILOT-1"
        folder = os.path.join(root, "computations")
    # DTU user
    elif user == 'mmilo':
        RH = False
        root = r"C:\phd\DTU-CFA-PILOT-1"
        folder = os.path.join(root, "computations")
    # RH user
    else: 
        RH = True
        if use_MI_data:
            root = r"G:\DTU-MI"
            folder = r"E:\DTUTeams\mml\MI_data"
        else: 
            root = r"H:\DTU-CFA-1"
            folder = r"E:\DTUTeams\mml\CFA_1"

    assert os.path.exists(root), f"Root path does not exist: {root}"

    return RH, root, folder

def get_name(split, case_id, sequential=False):
    name = f"{split.pseudonymized_id.iloc[case_id-1 if sequential else 0].strip()}_to_{split.pseudonymized_id.iloc[case_id].strip()}"
    return name

def load_CFA_data(split, root, folder, case_id=1, use_mask=True, sequential=False, verbose=False):
        
    source_path = os.path.join(root, "NIFTI", split.filename.iloc[case_id-1 if sequential else 0].strip())
    target_path = os.path.join(root, "NIFTI", split.filename.iloc[case_id].strip())
    
    if verbose:
        print(f"source path: {source_path}")
        print(f"target path: {target_path}")
    
    name = get_name(split, case_id, sequential)
    
    source_image = sitk.ReadImage(source_path)
    target_image = sitk.ReadImage(target_path)
    
    # convert values to [-1,1]
    source_im = sitk.GetArrayFromImage(source_image).astype(np.float32)
    s_min, s_max = np.quantile(source_im, [0.01, 0.99])
    source_im = (source_im - s_min) / (s_max - s_min) * 2 - 1
    source_im = np.clip(source_im, -1, 1)
    source_im = torch.FloatTensor(source_im)

    target_im = sitk.GetArrayFromImage(target_image).astype(np.float32)
    t_min, t_max = np.quantile(target_im, [0.01, 0.99])
    target_im = (target_im - t_min) / (t_max - t_min) * 2 - 1
    target_im = np.clip(target_im, -1, 1)
    target_im = torch.FloatTensor(target_im)
    
    if use_mask:
        mask_path = os.path.join(folder, split.pseudonymized_id.iloc[0].strip(), "segmentations", "total_seg/total_seg.nii.gz")
        source_mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(source_mask) > 0
        # dilate the mask with scipy
        mask = binary_dilation(mask, structure=np.ones((5,5,5)))
    else:
        mask = np.ones_like(source_im)
    
    return source_im, target_im, mask, source_image, name

def load_totalseg_data(split, root, folder, case_id=1, use_mask=True, sequential=False, use_sdf=False, verbose=False, sdf_max=20):
    
    
    source_path = os.path.join(folder, split.pseudonymized_id.iloc[case_id-1 if sequential else 0], "segmentations/total_seg/total_seg.nii.gz")
    target_path = os.path.join(folder, split.pseudonymized_id.iloc[case_id], "segmentations/total_seg/total_seg.nii.gz")
    
    if verbose:
        print(f"source path: {source_path}")
        print(f"target path: {target_path}")
    
    name = get_name(split, case_id, sequential)
    
    source_image = sitk.ReadImage(source_path)
    target_image = sitk.ReadImage(target_path)
    
    
    if use_sdf:

        source_im = sitk.GetArrayFromImage(source_image==5)
        source_sdf = -edt.sdf(source_im, 
                        anisotropy=source_image.GetSpacing()[::-1], 
                        parallel=8 # number of threads, <= 0 sets to num cpu
                        )
        source_im = torch.FloatTensor(source_sdf)
        source_im[source_im > sdf_max] = sdf_max
        source_im[source_im > 0] /= sdf_max
        source_im[source_im < 0] /= abs(source_im.min())

        target_im = sitk.GetArrayFromImage(target_image==5)
        target_sdf = -edt.sdf(target_im, 
                        anisotropy=target_image.GetSpacing()[::-1], 
                        parallel=8 # number of threads, <= 0 sets to num cpu
                        )
        target_im = torch.FloatTensor(target_sdf)
        target_im[target_im > sdf_max] = sdf_max
        target_im[target_im > 0] /= sdf_max
        target_im[target_im < 0] /= abs(target_im.min())

    else:
        # convert values to [-1,1]
        source_im = sitk.GetArrayFromImage(source_image).astype(np.float32)
        # s_min, s_max = source_im.min(), source_im.max()
        # source_im = (source_im - s_min) / (s_max - s_min) * 2 - 1
        source_im = torch.FloatTensor(source_im)

        target_im = sitk.GetArrayFromImage(target_image).astype(np.float32)
        # t_min, t_max = target_im.min(), target_im.max()
        # target_im = (target_im - t_min) / (t_max - t_min) * 2 - 1
        target_im = torch.FloatTensor(target_im)
    
    if use_mask:
        mask = sitk.GetArrayFromImage(source_image) > 0
        # dilate the mask with scipy
        mask = binary_dilation(mask, structure=np.ones((5,5,5)))
    else:
        mask = np.ones_like(source_im)
    
    return source_im, target_im, mask, source_image, name
    
def dice_score(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background
    elif isinstance(labels, int):
        labels = [labels]
        
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem)
    else:
        return (dicem, labels)
    
    
    
def get_bounding_box(reference):
    # Get the origin, size and spacing of the image
    origin = reference.GetOrigin()
    size = reference.GetSize()
    spacing = reference.GetSpacing()
    direction = reference.GetDirection()[::4]
    
    # Calculate the physical size of the image
    physical_size = [size[i]*spacing[i] for i in range(len(size))]
    
    # Calculate the bounding box
    direction = reference.GetDirection()[0::4]
    physical_size = [sz*sp for sz, sp in zip(size, spacing)]
    bounding_box = np.array([(o, o + d*s) for o, d, s in zip(origin, direction, physical_size)])
    bounding_box.sort(1)
    
    return bounding_box

def scale_points_from_reference_to_1_1(points, reference):

    bounding_box = get_bounding_box(reference)
    
    # Scale the points to the range [-1, 1]
    min_vals = bounding_box[:, 0]
    max_vals = bounding_box[:, 1]
    scaled_points = 2 * ((points - min_vals) / (max_vals - min_vals)) - 1
    return scaled_points

def scale_points_from_1_1_to_reference(scaled_points, reference):
    
    bounding_box = get_bounding_box(reference)
    
    # Scale the points back to the physical space
    min_vals = bounding_box[:, 0]
    max_vals = bounding_box[:, 1]
    points = ((scaled_points + 1) / 2) * (max_vals - min_vals) + min_vals
    return points