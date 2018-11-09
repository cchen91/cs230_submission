import os
import numpy as np
import nibabel as nib

pathNifti = "/data2/yeom/ky_fetal/Stanford_segmentations/"
pathDicom = "/data2/yeom/ky_fetal/Stanford_clean_series/"
pathDes = '/data2/yeom/ky_fetal/cs230_data/dir_list/'

index_ax1 = []
index_ax2 = []
index_cor1 = []
index_cor2 = []
index_sag1 = []
index_sag2 = []

for dirName, subdirList, fileList in os.walk(pathNifti):
    for filename in fileList:
        index_temp = [int(s) for s in filename.split("_") if s.isdigit()]
        if "_cor_" in filename:
            index_cor1.append(str(index_temp[0]))
            index_cor2.append(str(index_temp[1]))
        elif "_ax_" in filename:
            index_ax1.append(str(index_temp[0]))
            index_ax2.append(str(index_temp[1]))
        elif "_sag_" in filename:
            index_sag1.append(str(index_temp[0]))
            index_sag2.append(str(index_temp[1]))

pathDicom_ax = []
pathDicom_cor = []
pathDicom_sag = []
pathNifti_ax = []
pathNifti_cor = []
pathNifti_sag = []
num_ax=[]
num_cor=[]
num_sag=[]

for i in range(len(index_ax1)):
    pathD = pathDicom+index_ax1[i]+"/ax_"+index_ax2[i]+"/"
    pathN = pathNifti+index_ax1[i]+"_ax_"+index_ax2[i]+"_stanford_nml_seg.nii.gz"
    nifti = nib.load(pathN)
    nifti_shape = nifti.get_fdata().shape
    if nifti_shape[0] == 512 and i != 105 and i != 261 and i != 42 and i!= 105:
        pathDicom_ax.append(pathD)
        pathNifti_ax.append(pathN)

for i in range(len(index_cor1)):
    pathD = pathDicom+index_cor1[i]+"/cor_"+index_cor2[i]+"/"
    pathN = pathNifti+index_cor1[i]+"_cor_"+index_cor2[i]+"_stanford_nml_seg.nii.gz"
    nifti = nib.load(pathN)
    nifti_shape = nifti.get_fdata().shape
    if nifti_shape[0] == 512 and i != 5 and i != 29 and i != 83:
        pathDicom_cor.append(pathD)
        pathNifti_cor.append(pathN)

for i in range(len(index_sag1)):
    pathD = pathDicom+index_sag1[i]+"/sag_"+index_sag2[i]+"/"
    pathN = pathNifti+index_sag1[i]+"_sag_"+index_sag2[i]+"_stanford_nml_seg.nii.gz"
    nifti = nib.load(pathN)
    nifti_shape = nifti.get_fdata().shape
    if nifti_shape[0] == 512 and i != 32 and i != 98 and i != 133 and i!= 174:
        pathDicom_sag.append(pathD)
        pathNifti_sag.append(pathN)

np.savetxt(pathDes+'dicomPath_ax.txt', pathDicom_ax, delimiter = " ", fmt = "%s")
np.savetxt(pathDes+'dicomPath_cor.txt', pathDicom_cor, delimiter = " ", fmt = "%s")
np.savetxt(pathDes+'dicomPath_sag.txt', pathDicom_sag, delimiter = " ", fmt = "%s")
np.savetxt(pathDes+'niftiPath_ax.txt', pathNifti_ax, delimiter = " ", fmt = "%s")
np.savetxt(pathDes+'niftiPath_cor.txt', pathNifti_cor, delimiter = " ", fmt = "%s")
np.savetxt(pathDes+'niftiPath_sag.txt', pathNifti_sag, delimiter = " ", fmt = "%s")
