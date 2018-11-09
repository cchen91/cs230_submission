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

dicomDir = np.loadtxt(dirListPath+'dicomPath_sag.txt', dtype=str, delimiter = " ")
niftiDir = np.loadtxt(dirListPath+'niftiPath_sag.txt', dtype=str, delimiter = " ")

num_patient = len(niftiDir)

for i in range(num_patient):
    pathtmp = desPath+'sag_'+str(i+1)+'/'
    os.mkdir(pathtmp)
    y = nib.load(niftiDir[i]).get_data()
    y = np.rollaxis(y, 2, 0) #Moving number of images to axis 0
    y = y.transpose(0, 2, 1) #Transposing each image
    nm = y.shape[0]
    y = y.reshape((nm, nh, nw, 1))
    x = np.zeros((nm, nh, nw, 1))
    yalt = np.zeros((nm, nh, nw, 1))
    z = np.zeros((nh, nw, 3))
    zalt = np.zeros((nh, nw, 3))
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dicomDir[i]):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName,filename))
    for j in range(nm):
        x[j, ..., 0] = pydicom.read_file(lstFilesDCM[j]).pixel_array
    x = x/x.max()
    for j in range(nm):
        yalt[j] = y[nm-j-1]
        z[..., 0] = x[j, ..., 0]
        z[..., 2] = y[j, ..., 0]*0.7
        zalt[..., 0] = x[j, ..., 0]
        zalt[..., 2] = yalt[j, ..., 0]*0.7
        mpl.image.imsave(pathtmp+'z_'+str(j+1)+'.png', z)
        mpl.image.imsave(pathtmp+'zalt_'+str(j+1)+'.png', zalt)
    np.save(pathtmp+'x_sag_'+str(i+1)+'.npy', x)
    np.save(pathtmp+'y_sag_'+str(i+1)+'.npy', y)
    np.save(pathtmp+'yalt_sag_'+str(i+1)+'.npy', yalt)
    np.savetxt(pathtmp+'dicomList.txt', lstFilesDCM, delimiter = " ", fmt = "%s")


