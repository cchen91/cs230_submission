import os
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
import matplotlib as mpl

nh = 512
nw = 512
nptr = 231
npdev = 58

srcPath = "/data2/yeom/ky_fetal/cs230_data/raw/"
desPath = '/data2/yeom/ky_fetal/cs230_data/train/ax/'
xtr = []
ytr = []
xdev = []
ydev = []
order = np.loadtxt(srcPath+'ax_order.txt', dtype=str, delimiter = " ")

for i in range(npdev):
    index = i+1
    tmppath = srcPath+'ax_'+str(index)+'/'
    xtmp=np.load(tmppath+'x_ax_'+str(index)+'.npy')
    if order[index-1] == 'Y':
        ytmp = np.load(tmppath+'y_ax_'+str(index)+'.npy')
    else:
        ytmp = np.load(tmppath+'yalt_ax_'+str(index)+'.npy')
    nmtmp = xtmp.shape[0]
    for j in range(nmtmp):
        xdev.append(xtmp[j])
        ydev.append(ytmp[j])

np.save(desPath+'x_dev'+'.npy', np.asarray(xdev))
np.save(desPath+'y_dev'+'.npy', np.asarray(ydev))

for i in range(nptr):
    index = npdev+i+1
    tmppath = srcPath+'ax_'+str(index)+'/'
    xtmp=np.load(tmppath+'x_ax_'+str(index)+'.npy')
    if order[index-1] == 'Y':
        ytmp = np.load(tmppath+'y_ax_'+str(index)+'.npy')
    else:
        ytmp = np.load(tmppath+'yalt_ax_'+str(index)+'.npy')
    nmtmp = xtmp.shape[0]
    for j in range(nmtmp):
        xtr.append(xtmp[j])
        ytr.append(ytmp[j])
np.save(desPath+'x_tr'+'.npy', np.asarray(xtr))
np.save(desPath+'y_tr'+'.npy', np.asarray(ytr))
