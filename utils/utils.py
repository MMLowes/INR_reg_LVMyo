import numpy as np
import os
import glob
import pandas as pd
import re


def read_excel(root):
    """
    Reads the metadata excel file and returns CFA series for each patient.
    input:
        root: path to the patient folder (e.g. r"H:\DTU-CFA-1")
    output:
        list of dataframes with CFA series for each patient
    
    """
    file_name = os.path.join(root, "Metadata", os.path.basename(root)+"_meta_data.csv")
    df = pd.read_csv(file_name)
    
    # clean the dataframe
    df.rename(columns={n:n.strip() for n in df.columns}, inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.sort_values(by="filename")
    
    # filter the dataframe
    df = df[(df.SeriesDescription.str.contains("2.000 CE")  |
            df.SeriesDescription.str.contains("2.0 CE"  )   |
            df.SeriesDescription.str.contains("3.000 CE") ) &
            df.SeriesDescription.str.contains("%")          ]        
    vals = []
    
    # add percentage column
    for val in df.SeriesDescription:
        vals.append(int( re.search(r"\d+%",val).group().replace("%","")) )
    df["percentage"] = vals

    splits = []
    
    # loop over each patient and find the CFA series
    for idx in df.patient_id.unique():
        split = df[df.patient_id==idx].sort_values(by="percentage")
        
        # if 20 scans are from the same patient and all the files are present - add series
        if len(split) == 20 and \
            all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in split.filename]) and \
            (split.percentage==np.arange(0,100,5)).all(): 
            
            splits.append(split)
            
        # not a whole CFA scan
        elif len(split) < 20:
            continue
        
        # if more than 20 scans are present - find the series with the most slices and the same spacing
        elif len(split) > 20:
            for s in np.sort(split.n_slices.unique())[::-1]:
                sub_split = split[split.n_slices==s].copy()
                sub_split['spacing']=sub_split.PixelSpacing.str.extract(r'\[([\d.]+)\s')
                sub_split = sub_split[sub_split['spacing']==sub_split['spacing'].mode().iloc[0]]
                
                # if 20 scans have the same spacing, n_slices and all the files are present - add series
                if len(sub_split)==20 and \
                    all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in sub_split.filename]) and \
                    (sub_split.percentage==np.arange(0,100,5)).all(): 

                    splits.append(sub_split)
                    # only add one series per patient
                    break

    return splits 

def create_split_from_numbered_files(root,):
    """
    Create a split from a folder with numbered files.
    input:
        root: path to the patient folder (e.g. r"H:\DTU-CFA-1")
        n_files: number of files to include in the split
    output:
        dataframe with the split
    """
    files = glob.glob(os.path.join(root, "NIFTI", "*.nii.gz"))
    files = sorted(files, key=lambda x: int(re.search(r"\d+",os.path.basename(x)).group()))
    split = pd.DataFrame({"filename":[os.path.basename(f) for f in files]})
    # split["patient_id"] = os.path.basename(root)
    split["pseudonymized_id"] = split["filename"].apply(lambda x: x.split('.')[0])
    split["percentage"] = np.arange(0,100,100/len(files))
    return split
