import os
import numpy as np
import nibabel as nib
import pydicom

index_ax1 = []
index_ax2 = []
index_cor1 = []
index_cor2 = []
index_sag1 = []
index_sag2 = []

pathNifti = "/data2/yeom/ky_fetal/Stanford_segmentations/"

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

pathDicom = "/data2/yeom/ky_fetal/Stanford_clean_series/"

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
	img = nib.load(pathNifti_ax[i])
	img_shape = img.get_fdata().shape
	print(str(i))
	lstFilesDCM = []  # create an empty list
	for dirName, subdirList, fileList in os.walk(pathDicom_ax[i]):
		for filename in fileList:
			lstFilesDCM.append(os.path.join(dirName,filename))
	if img_shape[0] == 512 and i != 105 and i != 261:
		nifti_ax.append(img.get_fdata())
		dicom_temp = np.zeros(img_shape)
		for ii in range(img_shape[2]):
			ds = pydicom.read_file(lstFilesDCM[ii])
			dicom_temp[:, :, ii] = ds.pixel_array
		dicom_ax.append(dicom_temp)

for i in range(len(index_cor1)):
	pathDicom_cor.append(pathDicom+index_cor1[i]+"/cor_"+index_cor2[i]+"/")
	pathNifti_cor.append(pathNifti+index_cor1[i]+"_cor_"+index_cor2[i]+"_stanford_nml_seg.nii.gz")
	img = nib.load(pathNifti_cor[i])
	if img.get_fdata().shape[0] == 512:
		nifti_cor.append(img.get_fdata())

for i in range(len(index_sag1)):
	pathDicom_sag.append(pathDicom+index_sag1[i]+"/sag_"+index_sag2[i]+"/")
	pathNifti_sag.append(pathNifti+index_sag1[i]+"_sag_"+index_sag2[i]+"_stanford_nml_seg.nii.gz")
	img = nib.load(pathNifti_sag[i])
	if img.get_fdata().shape[0] == 512:
		nifti_sag.append(img.get_fdata())


