import os
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
import matplotlib as mpl

nh = 512
nw = 512

dirListPath = "/data2/yeom/ky_fetal/cs230_data/dir_list/"
desPath = '/data2/yeom/ky_fetal/cs230_data/raw/'

dicomDir = np.loadtxt(dirListPath+'dicomPath_ax.txt', dtype=str, delimiter = " ")
niftiDir = np.loadtxt(dirListPath+'niftiPath_ax.txt', dtype=str, delimiter = " ")

num_patient = len(niftiDir)
dicom_raw = [[] for _ in range(num_patient)]
nifti_raw = [[] for _ in range(num_patient)]
num_list = np.zeros((num_patient, 1))
for i in range(num_patient):
    nifti = nib.load(niftiDir[i])
    nifti_shape = nifti.get_fdata().shape
    num_list[i]=nifti_shape[2]
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dicomDir[i]):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))
    for j in range(nifti_shape[2]):
        nifti_raw[i].append(nifti.get_fdata()[:, :, nifti_shape[2]-j-1]) # label
        ds = pydicom.read_file(lstFilesDCM[j])
        dicom_raw[i].append(ds.pixel_array)
#np.savetxt(desPath+'numindex_ax.txt', num_list, delimiter = " ", fmt = "%s")

num_list=num_list.astype(int)
nm = np.sum(num_list)
x = np.zeros((nm, nh, nw, 1))
y = np.zeros((nm, nh, nw, 1))
xtemp = np.zeros((nh, nw))
ytemp = np.zeros((nh, nw))
z = np.zeros((nh, nw, 3))
for i in range(num_patient):
    for j in range(num_list[i, 0]):
        nbase = 0
        if i != 0:
            nbase = int(np.sum(num_list[:i-1]))
        xtemp = dicom_raw[i][j]
        xtemp = xtemp/xtemp.max()
        ytemp = nifti_raw[i][j].T
        x[nbase+j, ..., 0] = xtemp
        y[nbase+j, ..., 0] = ytemp
        z[..., 0] = xtemp
        z[..., 2] = ytemp
        mpl.image.imsave(desPath+'zax_'+str(i+1)+'_'+str(j+1)+'.png', z)

#np.save(pathDes+"x_ax.npy", x_ax)
#np.save(pathDes+"y_ax.npy", y_ax)

