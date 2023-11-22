# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.stats import spearmanr, kendalltau, zscore
import tensorflow as tf
import pickle as pkl
import os, requests, tarfile
import glob
import h5py
import pandas as pd
import json
import nibabel as nib
import copy
import re
import math
import PIL
from datetime import datetime
import cv2
import sys
from pathlib import Path
from scipy.io import loadmat
import scipy.io as spy


path = '/home/mor/ECoG/Data_4_IDC_collab/'
path_imported = '/home/mor/ECoG/imported_data/'
path_for_save = '/home/mor/ECoG/subjects_data/'
signal_dir = "segMats_rawHFA_smooth50ms"
stim_dir = "stimuli (3 task versions collapsed)/"
subs_metadata = pd.read_csv(path + "elecSummary_allVisual_anatInfo.csv")
brain_regions = ['V1', 'V2']


# +
all_resp_files = np.unique(os.listdir(path + signal_dir))
electrode_for_task_dict = {}
num_elec_task_dict = {}
matfiles_for_tasks_dict = {}
num_sub_task = {}

pattern_task = r"([^_]+\.mat)$"
all_tasks = np.unique([re.search(pattern_task, os.path.basename(str(filename))).group() for filename in all_resp_files])
# -

all_resp_files


# +
def ms_to_ind(ms):
    """
    Convert from miliseconds to ECoG signal sample index
    """
    return int((ms + 500)/2)

def normalize(signal, stime, etime, GBL = None, zscore_dict = None):
    """
    getting the ECoG signal and applying a normalization method as requested. methods can be: GBL, zscore, none
    signal: the ECoG signal
     
    stime: from which ms to sample
    etime: until which ms to sample
    GBL: is GBL option is chosen, the function caller need to provide the GBL value of this signal file
    """
    
    if GBL:
        return (np.mean(signal[:, ms_to_ind(stime):ms_to_ind(etime)]) - GBL)/GBL
    elif zscore_dict:
        return (np.mean(signal[:, ms_to_ind(stime):ms_to_ind(etime)]) - zscore_dict['mu'])/zscore_dict['sd']
    else:
        return np.mean(signal[:, ms_to_ind(stime):ms_to_ind(etime)])     


# -

def filter_by_subject_and_electrode(filenames, filter_list):
    pattern = r"(LIJ\d+)(?:_.*|)_e(\d+)_.*\.mat"
    files_to_work = []
    electrode_for_task = []
    for filename in filenames:
        findpattern = re.search(pattern, os.path.basename(str(filename)))
        try:
            if [findpattern.group(1), int(findpattern.group(2))] in filter_list:
                files_to_work.append(os.path.basename(str(filename)))
                if [findpattern.group(1), int(findpattern.group(2))] not in electrode_for_task:
                    electrode_for_task.append([findpattern.group(1), int(findpattern.group(2))] )
        except AttributeError:
            print(os.path.basename(str(filename)))
    return files_to_work, electrode_for_task


for br_reg in brain_regions:
    list_of_e = subs_metadata[['subjNames', 'elecNums']].loc[(subs_metadata["brodmann_bestLabel"] == br_reg) &
                                                             (subs_metadata["brodmann_dist2bestLabel"] < 10)].values.tolist()
    electrode_for_task_dict[br_reg] = {}
    num_elec_task_dict[br_reg] = {}
    matfiles_for_tasks_dict[br_reg] = {}
    num_sub_task[br_reg] = {}
    for task in all_tasks:
        matfiles = list(Path(path + signal_dir).glob("*" + task))
        matfiles_for_tasks_dict[br_reg][task], electrode_for_task_dict[br_reg][task] = filter_by_subject_and_electrode(matfiles, list_of_e)
        num_elec_task_dict[br_reg][task] = len(electrode_for_task_dict[br_reg][task])
        num_sub_task[br_reg][task] = {}
        for item in electrode_for_task_dict[br_reg][task]:
            subject = item[0]
            if subject in num_sub_task[br_reg][task]:
                num_sub_task[br_reg][task][subject] += 1
            else:
                num_sub_task[br_reg][task][subject] = 1

len(electrode_for_task_dict['V2']['eventRelatedNatural.mat'])

GBL_dict = {}
zscore_dict = {}
for br_reg in brain_regions:
    GBL_dict[br_reg] = {}
    zscore_dict[br_reg] = {}
    elecs = electrode_for_task_dict[br_reg]['eventRelatedNatural.mat']
    elecs = [str(elec[0]) +"_e"+ str(elec[1]) for elec in elecs]
    for elec in elecs:
        zscore_dict[br_reg][elec] = {}
        filenames = [filename for filename in matfiles_for_tasks_dict[br_reg]['eventRelatedNatural.mat'] if filename.startswith(elec)]
        files = [loadmat(path + signal_dir + "/" + fname)['segMat'] for fname in filenames]
        all_data = np.concatenate(files, axis=0)
        GBL_dict[br_reg][elec] = np.mean(all_data[:,150:250])
        zscore_dict[br_reg][elec]['mu'] = np.mean(all_data)
        zscore_dict[br_reg][elec]['sd'] = np.std(all_data)



zscore_dict

a = 1
if a:
    print(1)

# +
### Create DF for all files
tasks_types = ['eventRelatedNatural.mat']

# Parse each filename
df_rois = {}
images = []
RDMv = {}

for br_reg in brain_regions:
    # Empty lists to store extracted data
    subnums = []
    electrodes = []
    tasks = []
    filenames_list = []
    imagenames = []
    brain_region = []
    
    for task in tasks_types:
        filenames = matfiles_for_tasks_dict[br_reg][task]
        for filename in filenames:
            match = re.match(r"^((LIJ\d+_imp\d+)|(LIJ\d+))_(e\d+)_(\w+?\d+?)_(eventRelatedNatural)\.mat$",
                             filename)
            if match:
                brain_region.append(br_reg)
                subnums.append(match.group(1))  # Captures either format of subnum
                electrodes.append(match.group(4))
                tasks.append(match.group(6))
                filenames_list.append(filename)
                imagenames.append(match.group(5))

    # Create a pandas DataFrame
    df_data = pd.DataFrame({
        'subnum': subnums,
        'electrode': electrodes,
        'brainregion': brain_region,
        'task': tasks,
        'imagename': imagenames,
        'filename': filenames_list
    })
    
    df_data['value'] = 1
    df_data['df_elec_sub'] = df_data["subnum"] + "_" + df_data["electrode"]
    
    for idx in range (df_data.shape[0]):
        fname = df_data.iloc[idx]["filename"]
        file = loadmat(path + signal_dir + "/" + fname)['segMat']
        GBL = GBL_dict[br_reg][df_data.loc[1, 'df_elec_sub']]
        zscore_elec = zscore_dict[br_reg][df_data.loc[1, 'df_elec_sub']]
#         df_data.loc[idx, 'value'] = (np.mean(file[:, 275:425]) - GBL)/GBL
#         df_data.loc[idx, 'value'] = np.mean(file[:, 250:425])
        df_data.loc[idx, 'value'] =  normalize(file, 50, 150, GBL=GBL) # change normalization type
    pivot_df = df_data.pivot_table(index='df_elec_sub', columns='imagename', values='value', aggfunc='sum', fill_value=0)
    
    # Step 1: Filter rows without zero values
    pivot_df_1 = pivot_df[~(pivot_df == 0).any(axis=1)]

    # Step 2: Filter columns without zero values
    pivot_df_2 = pivot_df_1.loc[:, ~(pivot_df_1 == 0).any(axis=0)]
    
    images = "VS13_" + pivot_df_2.columns + "_stimuli_EventRelatedNatural.png"
    RDMv[br_reg] = 1 - np.corrcoef(pivot_df_2.T)

    df_rois[br_reg] = pivot_df_2
# -

all_images = np.array([plt.imread(path +"stimuli/"+ image) for image in images])

pd.set_option('display.max_rows', df_data.shape[0]+1)
df_data.loc[1, 'df_elec_sub']

pd.set_option('display.max_rows', df_rois['V2'].shape[0]+1)
display(df_rois['V2'])

pd.set_option('display.max_rows', df_data.shape[0]+1)
df_data

# +
with open(path_for_save + 'RDMv_GBL_dist10_100ms.pkl', 'wb') as handle:
    pkl.dump(RDMv, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print("saved RDMv!")
    
with open(path_for_save + 'all_images.pkl', 'wb') as handle:
    pkl.dump(all_images, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print("saved all_images!")
# -

GBL_dict
