import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
import torch
import vtk
from skimage import morphology
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from models import models
from utils import general, utils


def run_IDIR_on_CFA(current_split, root, folder, sequential=True, exist_ok=True, sdf_alpha=0.0):
    """
    Run the IDIR (Implicit Deep Image Registration) algorithm on CFA (Cardiac Fibrosis Atlas) dataset.

    Args:
        current_split (dataframe): The current split object, containing the patient ID, filenames and percentage.
        root (str): The root directory for NIFTI files.
        folder (str): The folder directory containing totalsegmetator segmentations in subfolders for each patient.
        sequential (bool, optional): Whether to use sequential mode. Defaults to True.
        exist_ok (bool, optional): Whether to overwrite existing files. Defaults to True.
        sdf_alpha (float, optional): The alpha value for SDF weight in the loss. Defaults to 0.0.

    Returns:
        None
    """

    kwargs = {}
    kwargs["verbose"] = True
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["jacobian_symmetric"] = True
    kwargs["bending_regularization"] = False
    kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"] = os.path.join(folder, f"IDIR_ncc_alpha_{sdf_alpha}{'_seq' if sequential else ''}")
    kwargs["batch_size"] = 10_000
    kwargs["epochs"] = 2000
    kwargs["lr"] = 1e-5
    kwargs["layers"] = [3, 256, 256, 256, 256, 256, 3]
    kwargs["loss_function"] = "ncc"
    kwargs["4d_input"] = (0.0 < sdf_alpha < 1.0) 
    kwargs["sdf_alpha"] = sdf_alpha

    # alpha=0 -> only hu
    # alpha=1 -> only sdf
    if sdf_alpha == 1.0:
        use_sdf = True
    elif sdf_alpha == 0.0:
        use_sdf = False

    os.makedirs(kwargs["save_folder"], exist_ok=True)

    source_path = os.path.join(folder, current_split.pseudonymized_id.iloc[0], "segmentations/total_seg/total_seg.nii.gz")
    source_seg = sitk.ReadImage(source_path)
    s_seg = sitk.GetArrayFromImage(source_seg)
    sitk.WriteImage(source_seg, f"{kwargs['save_folder']}/moved_totalseg_{current_split.percentage.iloc[0]:02d}.nii.gz")
    
    dice_scores = {}

    for i in range(1, 20):
        source_image, target_image, mask, reference, name = general.load_CFA_data(current_split, root, folder, i, use_mask=True, sequential=sequential)
        source_sdf,   target_sdf, _, _, _ = general.load_totalseg_data(current_split, root, folder, i, use_mask=True, sequential=sequential, use_sdf=True)
        kwargs["mask"] = mask
        
        if kwargs["4d_input"]:
            source_input = torch.stack([source_image, source_sdf], dim=-1)
            target_input = torch.stack([target_image, target_sdf], dim=-1)
        elif use_sdf:
            source_input = source_sdf
            target_input = target_sdf
        else:
            source_input = source_image
            target_input = target_image
            
        ImpReg = models.ImplicitRegistrator(source_input, target_input, **kwargs)
        
        if os.path.exists(f"{kwargs['save_folder']}/network_{current_split.percentage.iloc[i]:02d}_{name}.pth") and exist_ok:
            ImpReg.load_network(f"{kwargs['save_folder']}/network_{current_split.percentage.iloc[i]:02d}_{name}.pth")
            print(f"Loaded network from {kwargs['save_folder']}/network_{current_split.percentage.iloc[i]:02d}_{name}.pth")
        else:
            ImpReg.fit()
            ImpReg.save_network(f"{kwargs['save_folder']}/network_{current_split.percentage.iloc[i]:02d}_{name}.pth")
        
        moved = ImpReg.transform_volume(source_image.shape, source_image)
        moved_image = sitk.GetImageFromArray(moved)
        moved_image.CopyInformation(reference)
        sitk.WriteImage(moved_image, f"{kwargs['save_folder']}/moved_image_{current_split.percentage.iloc[i]:02d}_{name}.nii.gz")
        

        if sequential:
            source_path = os.path.join(folder, current_split.pseudonymized_id.iloc[i-1], "segmentations/total_seg/total_seg.nii.gz")
            source_seg = sitk.ReadImage(source_path)
            s_seg = sitk.GetArrayFromImage(source_seg)
            
        target_path = os.path.join(folder, current_split.pseudonymized_id.iloc[i], "segmentations/total_seg/total_seg.nii.gz")
        target_seg = sitk.ReadImage(target_path)
        t_seg = sitk.GetArrayFromImage(target_seg)

        ImpReg.network.eval()
        transformed_seg = ImpReg.transform_volume(s_seg.shape, s_seg)
        transformed_seg = transformed_seg.astype(int)
        dice, dice_lvm = general.dice_score(t_seg, transformed_seg), general.dice_score(t_seg, transformed_seg, 5)
        dice_scores[name] = [dice, dice_lvm]
        
        tr_seg = sitk.GetImageFromArray(transformed_seg)
        tr_seg.CopyInformation(reference)

        sitk.WriteImage(tr_seg, f"{kwargs['save_folder']}/moved_totalseg_{current_split.percentage.iloc[i]:02d}_{name}.nii.gz")


    print("Dice scores:")
    for key, value in dice_scores.items():
        print(f"{key}: {value[0]:.4f} -- {value[1]:.4f}")
        
    #save json
    with open(os.path.join(kwargs['save_folder'], "dice_scores.json"), "w") as f:
        json.dump(dice_scores, f)
   
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Run IDIR on CFA")
    # parser.add_argument("--root", type=str, help="Root directory for NIFTI files")
    parser.add_argument("--path", type=str, required=True, help="Path to the CFA dataset")
    parser.add_argument("--use_metadata", action="store_true", help="Whether the dataset has metadata")
    parser.add_argument("--sequential", action="store_true", help="Run in sequential mode")
    parser.add_argument("--sdf_alpha", type=float, default=0.0, help="SDF alpha value")
    parser.add_argument("--exist_ok", action="store_true", default=True, help="Overwrite existing files")

    
    args = parser.parse_args()
    
    if args.use_metadata:
        RH, root, home_folder = general.get_path_and_root()
        splits = utils.read_excel(root)
        split = splits[0 if RH else 1]
        folder = os.path.join(home_folder, split.patient_id.iloc[0])
    else:
        root = args.path
        folder = os.path.join(args.path, "computations")
        split = utils.create_split_from_numbered_files(root)

    
    run_IDIR_on_CFA(split, root, folder, sequential=args.sequential, exist_ok=args.exist_ok, sdf_alpha=args.sdf_alpha)
