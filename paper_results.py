
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import json
from collections import defaultdict
import seaborn as sns
import seg_metrics.seg_metrics as sg
import glob
import torch

from models import models
from utils import utils, general

def dice_results(splits, home_folder, root):
    scores = []
    for i in range(len(splits)):

            split = splits[i]

            folder = os.path.join(home_folder, split.patient_id.iloc[0])

            dice_scores = {}
            for sdf_alpha in [0,0.8,1.0]:
                for sequential in [False, True]:
                    save_folder = os.path.join(folder,  f"IDIR_ncc_alpha_{sdf_alpha}{'_seq' if sequential else ''}")
                    try:
                        with open(os.path.join(save_folder, "dice_scores.json"), "r") as f:
                            d = json.load(f)
                        dice_scores[(sdf_alpha, sequential)] = d
                        
                    except:
                        continue
            if len(dice_scores) == 6:
                scores.append(dice_scores)
                print(f"Loaded {split.patient_id.iloc[0]}")

    mean_scores = {'whole': defaultdict(list),
                'lvm': defaultdict(list)}

    for key in scores[0].keys():
        for i in range(len(scores)):
            values = np.array(list(scores[i][key].values()))
            mean_scores['whole'][key].append(values[:,0])
            mean_scores['lvm'][key].append(values[:,1])


    sns.set_theme('talk')
    sns.set_style('ticks')

    # repeat plot with error bars
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for key, value in mean_scores['lvm'].items():
        label = fr"$\alpha$: {key[0] if key[0]==0.8 else 1-key[0]:.1f}"
        vals = [1]+np.mean(value, axis=0).tolist()
        stds = [0]+np.std(value, axis=0).tolist()
        if key[1]:
            ax1.plot(vals, label=label)
            ax1.fill_between(np.arange(0,20), np.array(vals)-np.array(stds), np.array(vals)+np.array(stds), alpha=0.2)
        else:
            ax2.plot(vals, label=label)
            ax2.fill_between(np.arange(0,20), np.array(vals)-np.array(stds), np.array(vals)+np.array(stds), alpha=0.2)

    # ax1.set_title("Sequential - LVM")
    ax1.legend(loc='lower right')
    ax1.set_xlabel("Scan percentage")
    ax1.set_ylabel("Dice score")
    ax1.set_xticks(np.arange(0,20,2))
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=20.0))

    # ax2.set_title("Non-sequential - LVM")
    ax2.legend(loc='lower right')
    ax2.set_xlabel("Scan percentage")
    ax2.set_ylabel("Dice score")
    ax2.set_xticks(np.arange(0,20,2))
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=20.0))

    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax1.set_xlim(0, 19)
    ax2.set_xlim(0, 19)
    fig1.tight_layout(pad=1.5)
    fig2.tight_layout(pad=1.5)
    plt.show()
    table = [[],[]]
    for key, value in mean_scores['lvm'].items():
        if key[1]:
            table[0].append(np.mean(value))
            print(key[0])
        else:
            table[1].append(np.mean(value))
            print(key[0])

    for i in range(2):
        for j in range(3):
            print(f"{100*table[i][j]:.2f}", end=" & ")
        

    print(f"\nDice score from {len(scores)} patients")

def hausdorff_results(splits, home_folder, root):
    exists_ok = True
    use_hd95 = True

    mean_hd = {'whole': defaultdict(list),
                'lvm': defaultdict(list)}

    for j in range(len(splits)):
        split = splits[j]
        print(split.patient_id.iloc[0])
        folder = os.path.join(home_folder, split.patient_id.iloc[0])

        for sdf_alpha in [0, 0.8, 1.0]:
            for sequential in [True, False]:

                save_folder = os.path.join(folder, f"IDIR_ncc_alpha_{sdf_alpha}{'_seq' if sequential else ''}")
                try:
                    if os.path.isfile(f"{save_folder}/hausdorff.npy") and exists_ok:
                        haus = np.load(f"{save_folder}/hausdorff.npy")
                    else:
                        haus = np.zeros((20,4))
                        for i in range(1, 20):
                            seg_moved = glob.glob(save_folder+"/moved_totalseg*")[i]
                            seg_target = os.path.join(folder, split.pseudonymized_id.iloc[i], "segmentations", "total_seg", "total_seg.nii.gz")

                            labels = [0, 1, 2, 3, 4, 5, 7, 8]
                            metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                                    gdth_path=seg_target,
                                    pred_path=seg_moved,
                                    metrics=["hd","hd95"])

                            haus[i] = [np.mean(metrics[0]["hd"]), metrics[0]["hd"][4], np.mean(metrics[0]["hd95"]), metrics[0]["hd95"][4]]
                        np.save(f"{save_folder}/hausdorff.npy", haus)

                    if use_hd95:
                        mean_hd['whole'][(sdf_alpha, sequential)].append(haus[:,2])
                        mean_hd['lvm'][(sdf_alpha, sequential)].append(haus[:,3])
                    else:
                        mean_hd['whole'][(sdf_alpha, sequential)].append(haus[:,0])
                        mean_hd['lvm'][(sdf_alpha, sequential)].append(haus[:,1])
                    

                except IndexError:
                    print(f"IndexError: {split.patient_id.iloc[0]}")
                    continue

    table = [[],[]]
    for key, value in mean_hd['lvm'].items():
        if key[1]:
            table[0].append(np.mean(value))
            print(key[0])
        else:
            table[1].append(np.mean(value))
            print(key[0])

    for i in range(2):
        for j in range(3):
            print(f"{table[i][j]:.4f}", end=" & ")
    
    print(f"\nDice score from {len( mean_hd['lvm'])} patients")
    
def landmarks_results(split, folder, root):
    
    landmarks = []

    for i in range(len(split)):
        lm_path = os.path.join(folder, f"landmarks/Jonas_{split.percentage.iloc[i]:02d}.mrk.json")
        with open(lm_path, "r") as f:
            lm = json.load(f)
        points = [p['position'] for p in lm['markups'][0]['controlPoints']]
        landmarks.append(points)

    landmarks = np.array(landmarks)

    results = {}
    for sequential in [True, False]:
        for sdf_alpha in [0, 0.8, 1.0]:

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
            kwargs["4d_input"] = (sdf_alpha > 0.0 and sdf_alpha < 1.0)
            kwargs["sdf_alpha"] = sdf_alpha

            source_image, target_image, mask, reference, name = general.load_CFA_data(split, root, folder, i, use_mask=True, sequential=sequential)
            kwargs["mask"] = mask

            ImpReg = models.ImplicitRegistrator(torch.stack([source_image, source_image], dim=-1), torch.stack([target_image, target_image], dim=-1), **kwargs) if kwargs["4d_input"] else \
                     models.ImplicitRegistrator(source_image, target_image, **kwargs)
            

            tracked_points = np.zeros_like(landmarks)
            tracked_points[0] = landmarks[0]
            scaled_points = general.scale_points_from_reference_to_1_1(landmarks[0], reference)

            for i in range(1, len(split)):
                name = general.get_name(split, i, sequential)
                ImpReg.load_network(f"{kwargs['save_folder']}/network_{split.percentage.iloc[i]:02d}_{name}.pth")
                moved_points = ImpReg.transform_points(scaled_points)
                tracked_points[i] = general.scale_points_from_1_1_to_reference(moved_points, reference)
                scaled_points = moved_points.copy() if sequential else general.scale_points_from_reference_to_1_1(landmarks[i], reference)

            diffs = np.linalg.norm(tracked_points - landmarks, axis=-1)
            results[(sdf_alpha,sequential)] = [diffs, tracked_points]

    i = 3
    xlim = (-50,-30) #(25,50)
    ylim = (1740,1760) #(-15,10)

    for k,v in results.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(v[1][:,i,1], v[1][:,i,2], label=f"{k}", color='black')
        ax.set_title(k)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.axis('off')
        plt.show()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(landmarks[:,i,1], landmarks[:,i,2], label=f"{i+1}", color='black')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.axis('off')
    plt.show()

    # for k, v in results.items():
    #     print(k, np.mean(v[0]))
        
    table = [[],[]]
    for key, value in results.items():
        if key[1]:
            table[0].append(np.mean(value[0]))
        else:
            table[1].append(np.mean(value[0]))

    for i in range(2):
        for j in range(3):
            print(f"{table[i][j]:.4f}", end=" & ")
    


   
if __name__ == "__main__":

    
    RH, root, home_folder = general.get_path_and_root()
    splits = utils.read_excel(root)
    current_split = splits[0]
    folder = os.path.join(home_folder, current_split.patient_id.iloc[0])

    dice_results(splits, home_folder, root)
    
    hausdorff_results(splits, home_folder, root)
    
    landmarks_results(current_split, folder, root)