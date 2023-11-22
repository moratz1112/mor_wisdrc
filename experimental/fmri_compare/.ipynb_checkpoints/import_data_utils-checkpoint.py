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


path = '/home/mor/NDS_project/imported_data'
sub_path = path + "/sub"
num_beta = 750


class pic_edit():
    """
    This class should ease applying filters and edits to the cut squered pictue before they are being proccessed by the net. 
    It aims to contain all present and future potential dynamic properties of a picture.
    """
    def __init__(self, size = 224, blur = 1):
        self.size = size
        self.blur = blur
        
    def __repr__(self):
        return f'pic({self.size, self.blur})'
    
    def __hash__(self):
        return hash((self.size, self.blur))


# loading images data
def import_sub_stim_data(path:str, subnums:list, stim_data:str):
    """
    This function takes the stimulation meta data, which states which subject got which stim and when, and return a
    table with the meta data of the stims of the selected subject got, out of the first 1500 stimulations.
    
    path: str represent the project's files path
    subnums: a list of subject nums, so we filter only rows of stimulation that all the defined subject were experienced
    stim_data: stimulation data file name
    """
    stiminf = pd.read_pickle(path + stim_data)
    for subnum in subnums:
        snum = str(subnum)    
        stiminf = stiminf.loc[(stiminf['subject' + snum] == True) & (stiminf["subject" + snum + "_rep0"] <= 1500)]
    col_rep = ["subject" + str(snum) + "_rep0" for snum in subnums]
    substims = stiminf.filter(["cocoId", "cocoSplit","cropBox"] + col_rep).sort_values(by="subject" + snum + "_rep0")
    substims["zeronum"] =   (12 - substims['cocoId'].fillna('').astype(str).str.len())
    substims["path"] = (substims["cocoSplit"] +"/" + substims["cocoSplit"] + "/" + substims["zeronum"].apply(lambda x: x * '0') + (substims["cocoId"]).astype(str) + ".jpg")
    substims = substims.reset_index()
    return substims


def load_batas(subpath:str) -> (np.ndarray):
    """
    This function will load beta files for a subject path
    """
    betas_paths = sorted(glob.glob(subpath + "/betas_hdf/betas_session*"))
    try:
        betas = h5py.File(betas_paths[0], "r")["betas"]
    except IndexError:
        print('No beta files for this subject!')
    if len(betas_paths) > 1:
        for path in betas_paths[1:]:
            betas =  np.concatenate((betas, 
                    h5py.File(path, "r")["betas"]), axis=0)
    print("the size of the beta is: " + str(betas.shape) + " the dimensions are: trial * Voxel X * Voxel Y * Voxel Z")
    return betas


# ROI mappings extracting
def get_roi_vox_data(roi_voxels_fname:str, roi_dict_fname:str) -> (dict):
    """
       This function converts the voxels-ROI and ROInum-ROIname files to np.ndarray and a dictionary for convinience.
    """
    visROIS = nib.load(roi_voxels_fname)
    unique, counts = np.unique(visROIS.get_fdata(), return_counts=True)
    roi_count = dict(zip(unique, counts))
    print("collecting data from: " + re.search('sub\d/ROIs/(.+?).nii.gz',roi_voxels_fname).group(1))
    print ("relevant voxels: " + str(sum(value for key, value in roi_count.items() if key > 0)))
    print (roi_count)
    roi_legend = open(roi_dict_fname, "r")
    roi_len = roi_legend.read()
    reg_list = re.split('\t +\n|\n', roi_len)[:-1]
    reg_dict = {val[0]:val[2:] for val in reg_list}
    reg_dict[-1] = "unrelated"
    print("categories are: " + str(reg_dict))
    return {'mapping': np.transpose(visROIS.get_fdata(),(2,1,0)), 'legend': reg_dict}


def get_voxels_roi(collection:list, betas, pics_df, df_col):
    """
    This function will return a matrix all the stimului * all the voxels of a certain ROI 
    inputs:
        collection: list of two which is an object of the ROI dict output
        betas: the betas signal
    """
    voxels_ROIS = {}
    ROI_num = len(collection['legend'].keys())
    for i in range(1, ROI_num-1):
        ROI_reg_name = collection['legend'][str(i)]
        filterROI = (collection['mapping'] == i).astype(int)
        betas_reg_roi = np.multiply(betas, np.expand_dims(filterROI, axis=0))
        betas_reg_roi = betas_reg_roi.reshape((len(betas), np.prod(list(betas_reg_roi.shape[1:]))))
        betas_reg_roi = betas_reg_roi[:,~np.all(betas_reg_roi == 0, axis=0)]
        when_pics = list(pics_df[df_col]) # filtering out the beta reps when there was no pic presented
        when_pics = [when-1  for when in when_pics if when <= len(betas)]
        betas_reg_roi = betas_reg_roi[when_pics,:]
        betas_reg_roi = zscore(betas_reg_roi)
        voxels_ROIS[ROI_reg_name] = betas_reg_roi
        print("extracting voxels for: " + ROI_reg_name)
    return voxels_ROIS


def create_voxels_ROIs(subnum, beta, substims):
    """
    This function creates the Voxels ROIS names and values for a certain subjects
    subnum: subject number
    beta: the FMRI signal
    substims: The table with the meta data of the stims of the selected subject got, out of the first 1500 stimulations
    """
    subroi_path = "/home/mor/NDS_project/imported_data/sub" + str(subnum) + "/ROIs"
    roi_voxels_fnames = sorted(glob.glob(subroi_path +"/*.nii.gz"))
    roi_dict_fnames = sorted(glob.glob(subroi_path +"/*.mgz.ctab"))
    ROIs = {re.search(str(subnum) + '/ROIs/(.+?).nii.gz',vfname).group(1) : get_roi_vox_data(vfname, rfname) for vfname, rfname in zip(roi_voxels_fnames, roi_dict_fnames)}
    voxels_ROIs_full = {key : get_voxels_roi(ROIs[key], beta, substims, 'subject'+ str(subnum) +'_rep0') for key in ROIs.keys()}
    return voxels_ROIs_full


# pictures cutting
def cut_pic(im, cutbox):
    """
    This function gets the original image, and cuts it according to the matching cropBox given by the metadata table.
    the crop box used to cut the pictures to present them to the subject in the fMRI experiment 
    """
    im_len = im.shape[0]
    im_wid = im.shape[1]
    cut_im = np.ones(im.shape)*128
    cut_im = im[int(im_len*cutbox[0]):int(im_len*(1-cutbox[1])), int(im_wid*cutbox[2]):int(im_wid*(1-cutbox[3])),:]
    return cut_im.astype(int)


def cutpics(substims, path):
    """
    This function gets the pandas table of pictures, and cuts them as written.
    """
    ims_cuts = [1]*len(substims)
    for i in range(len(substims)):
        im_sub = plt.imread(path +"/"+ substims.loc[i]["path"])
        ims_cuts[i] = cut_pic(im_sub, substims.loc[i]["cropBox"])
    print("Done cutting pics")
    return(ims_cuts)


def create_RDM_fmri(voxels_ROIs: dict) -> (dict):
    """
    This function will get voxels ROIs signal data and will return a dictionary of the RDM of the data.
    """
    RDM_fmri = {}
    for area in voxels_ROIs.keys():
        RDM_fmri[area] = {}
        for roi in voxels_ROIs[area].keys():
            RDM_fmri[area][roi] = 1-np.corrcoef(voxels_ROIs[area][roi][:net_lim,:])
        print("Created RDM for: " + str(area))
    return RDM_fmri


def import_data_for_sub(subnum:int, path:str) -> (np.ndarray, dict): 
    """
    This function will get a subject number and return his relevent files
    output:
        cutims: an ndarray of all the subjects pictures
        voxels_ROIs_full: a dict with all ROIs
    """
    subpath = path + "/sub" + str(subnum)
    if not (os.path.isdir(subpath)):
        return "No directory for this subject"
    voxelspath =  glob.glob(subpath + "/voxels_ROIs_full*")
    voxels_ROIs_full = []
    cutims = []
    substims = import_sub_stim_data(path, [subnum], '/nsd_stim_info_merged.pkl')

    if voxelspath != []:
        with open (voxelspath[0], 'rb') as f:
            voxels_ROIs_full = pkl.load(f)
    else:
        betas = load_batas(subpath)
        voxels_ROIs_full = create_voxels_ROIs(subnum, betas, substims)
        with open(subpath + '/voxels_ROIs_full.pkl', 'wb') as handle:
            pkl.dump(voxels_ROIs_full, handle, protocol=pkl.HIGHEST_PROTOCOL)
            
    cutpath = glob.glob(subpath + "/cutims*")
    if cutpath != []:
        with open (cutpath[0], 'rb') as f:
            cutims = pkl.load(f)
    else:
        cutims = cutpics(substims, path)
        with open(subpath + '/cutims.pkl', 'wb') as handle:
            pkl.dump(cutims, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return cutims, voxels_ROIs_full
