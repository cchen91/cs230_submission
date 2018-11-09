import os
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt

pathNifti = "/data2/yeom/ky_fetal/Stanford_segmentations/"
pathDicom = "/data2/yeom/ky_fetal/Stanford_clean_series/"
pathDes = '/data2/yeom/ky_fetal/cs230_train/'

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
nifti_ax = []
nifti_cor = []
nifti_sag = []
dicom_ax = []
dicom_cor = []
dicom_sag = []

for i in range(len(index_ax1)):
	pathDicom_ax.append(pathDicom+index_ax1[i]+"/ax_"+index_ax2[i]+"/")
	pathNifti_ax.append(pathNifti+index_ax1[i]+"_ax_"+index_ax2[i]+"_stanford_nml_seg.nii.gz")
	nifti = nib.load(pathNifti_ax[i])
	nifti_shape = nifti.get_fdata().shape
	if nifti_shape[0] == 512 and i != 105 and i != 261 and i != 42 and i!= 105:
		lstFilesDCM = []  # create an empty list
		for dirName, subdirList, fileList in os.walk(pathDicom_ax[i]):
			for filename in fileList:
				lstFilesDCM.append(os.path.join(dirName,filename))
		for i in range(nifti_shape[2]):		
			nifti_ax.append(nifti.get_fdata()[:, :, i]) # label
			ds = pydicom.read_file(lstFilesDCM[i])
			dicom_ax.append(ds.pixel_array)

for i in range(len(index_cor1)):
	pathDicom_cor.append(pathDicom+index_cor1[i]+"/cor_"+index_cor2[i]+"/")
	pathNifti_cor.append(pathNifti+index_cor1[i]+"_cor_"+index_cor2[i]+"_stanford_nml_seg.nii.gz")
	nifti = nib.load(pathNifti_cor[i])
	nifti_shape = nifti.get_fdata().shape
	if nifti_shape[0] == 512 and i != 5 and i != 29 and i != 83:
		lstFilesDCM = []  # create an empty list
		for dirName, subdirList, fileList in os.walk(pathDicom_cor[i]):
			for filename in fileList:
				lstFilesDCM.append(os.path.join(dirName,filename))
		for i in range(nifti_shape[2]):
			nifti_cor.append(nifti.get_fdata()[:, :, i]) # label
			ds = pydicom.read_file(lstFilesDCM[i])
			dicom_cor.append(ds.pixel_array)

for i in range(len(index_sag1)):
	pathDicom_sag.append(pathDicom+index_sag1[i]+"/sag_"+index_sag2[i]+"/")
	pathNifti_sag.append(pathNifti+index_sag1[i]+"_sag_"+index_sag2[i]+"_stanford_nml_seg.nii.gz")
	nifti = nib.load(pathNifti_sag[i])
	nifti_shape = nifti.get_fdata().shape
	if nifti_shape[0] == 512 and i != 32 and i != 98 and i != 133 and i!= 174:
		lstFilesDCM = []  # create an empty list
		for dirName, subdirList, fileList in os.walk(pathDicom_sag[i]):
			for filename in fileList:
				lstFilesDCM.append(os.path.join(dirName,filename))
		for i in range(nifti_shape[2]):
			nifti_sag.append(nifti.get_fdata()[:, :, i]) # label
			ds = pydicom.read_file(lstFilesDCM[i])
			dicom_sag.append(ds.pixel_array)

size_ax = len(dicom_ax)
size_cor = len(dicom_cor)
size_sag = len(dicom_sag)

x_ax_2d = np.zeros((size_ax, 512, 512))
y_ax_2d = np.zeros((size_ax, 512, 512))
x_cor_2d = np.zeros((size_cor, 512, 512))
y_cor_2d = np.zeros((size_cor, 512, 512))
x_sag_2d = np.zeros((size_sag, 512, 512))
y_sag_2d = np.zeros((size_sag, 512, 512))

for i in range(size_ax):
	x_ax_2d[i, ...] = dicom_ax[i]
	y_ax_2d[i, ...] = nifti_ax[i].T

for i in range(size_cor):
	x_cor_2d[i, ...] = dicom_cor[i]
	y_cor_2d[i, ...] = nifti_cor[i].T

for i in range(size_sag):
	x_sag_2d[i, ...] = dicom_sag[i]
	y_sag_2d[i, ...] = nifti_sag[i].T

np.save(pathDes+"x_ax_2d.npy", x_ax_2d)
np.save(pathDes+"y_ax_2d.npy", y_ax_2d)
np.save(pathDes+"x_cor_2d.npy", x_cor_2d)
np.save(pathDes+"y_cor_2d.npy", y_cor_2d)
np.save(pathDes+"x_sag_2d.npy", x_sag_2d)
np.save(pathDes+"y_sag_2d.npy", y_sag_2d)
