import os
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

index = []
xtest_raw = []

path_src = "/data2/yeom/ky_fetal/Cincinnati/data/KlineStanford2"
path_des = "/data2/yeom/ky_fetal/cs230_test/"

for dirName, subdirList, fileList in os.walk(path_src):
    for filename in fileList:
        if ".dcm" in filename:
            index.append(os.path.join(path_src, dirName, filename))

nmraw = len(index)

for i in range(nmraw):
    ds = pydicom.dcmread(index[i])
    shape = ds.pixel_array.shape
    if shape[0] == 512 and len(shape) == 2:
        xtest_raw.append(ds.pixel_array)

nm = len(xtest_raw)
nh, nw = xtest_raw[0].shape
xte = np.zeros((nm, nh, nw, 1))

for i in range(nm):
    xte[i, ..., 0] = xtest_raw[i]
xte.max(axis = (1, 2))

xte.astype('float32')
xte = xte/xte.max(axis = (1, 2, 3), keepdims = True)
np.save(path_des+'xte.npy', xte)

for i in range(nm):
    mpl.image.imsave(path_des+'images/x_'+str(i+1)+'.png', xte[i, ..., 0], cmap = 'gray')
