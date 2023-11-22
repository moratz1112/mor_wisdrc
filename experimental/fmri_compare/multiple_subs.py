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

subs = [1,2]
from import_data_utils import *
cutims1, voxels_ROIs_full1 = import_data_for_sub(1, path)
cutims2, voxels_ROIs_full2 = import_data_for_sub(2, path)
mutual_stim = import_sub_stim_data(path, subs, '/nsd_stim_info_merged.pkl')
batch_size = 64
mutual_stim

sub1pic = import_sub_stim_data(path, [1], '/nsd_stim_info_merged.pkl')
mut_pic= np.asarray(mutual_stim['subject1_rep0'])
mut_pic_ind = np.asanyarray(sub1pic[sub1pic['subject1_rep0'].isin(list(mut_pic))].index)
mut_pic_ind

voxels_ROIs_full1_mut = {mainkey : {seckey : voxels_ROIs_full1[mainkey][seckey][mut_pic_ind] for seckey in voxels_ROIs_full1[mainkey].keys()} for mainkey in voxels_ROIs_full1.keys()}
voxels_ROIs_full2_mut = {mainkey : {seckey : voxels_ROIs_full2[mainkey][seckey][mut_pic_ind] for seckey in voxels_ROIs_full2[mainkey].keys()} for mainkey in voxels_ROIs_full2.keys()}

# +
net_lim = (len(mut_pic_ind) // batch_size) * batch_size


RDM_fmri_s1 = {}
for area in voxels_ROIs_full1_mut.keys():
    RDM_fmri_s1[area] = {}
    for roi in voxels_ROIs_full1_mut[area].keys():
        RDM_fmri_s1[area][roi] = 1-np.corrcoef(voxels_ROIs_full1_mut[area][roi][:net_lim,:])
    print("Created RDM for: " + str(area))
    
RDM_fmri_s2 = {}
for area in voxels_ROIs_full2_mut.keys():
    RDM_fmri_s2[area] = {}
    for roi in voxels_ROIs_full2_mut[area].keys():
        RDM_fmri_s2[area][roi] = 1-np.corrcoef(voxels_ROIs_full2_mut[area][roi][:net_lim,:])
    print("Created RDM for: " + str(area))

# corr = {}
# for area in RDM_fmri_s1.keys():
#     corr[area] = {}
#     for roi in RDM_fmri_s1[area].keys():
#         corr[area][roi] = np.corrcoef()

# +
def corr_list_RDM(RDM_fmri1, RDM_fmri2, corr_fun=spearmanr):
    """
    This function takes and all RDMS from FMRI voxels of 2 subjects and calculate correlations.
    """
    corr_dict = {}
    for brain_area in RDM_fmri1.keys():
        mask = np.triu(np.ones(RDM_fmri1[brain_area].shape,dtype=np.bool),1)
        corr_dict[brain_area] = []
        corr, pval = corr_fun(np.nan_to_num(RDM_fmri1[brain_area][mask]),np.nan_to_num(RDM_fmri2[brain_area][mask]))
        corr_dict[brain_area].append(round(corr, 3))
        print ("done correlations " + brain_area)
    return corr_dict

corr_dict_subjects = {key : corr_list_RDM(RDM_fmri_s1[key], RDM_fmri_s2[key]) for key in RDM_fmri_s1.keys()} 
corr_dict_subjects
# display(RDM_fmri_s1['ROIs/floc-bodies']['EBA'])
# -

filename = "compare"+"".join(["_" + str(sub) for sub in subs])
print(filename)
corr_dict_subjects

with open('/home/mor/NDS_project/results_data/subscompare/'+ filename + '.pkl' , 'wb') as handle:
        pkl.dump(corr_dict_subjects, handle, protocol=pkl.HIGHEST_PROTOCOL)

