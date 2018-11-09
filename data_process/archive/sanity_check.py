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

f = open("error_list.txt", "a")

for i in range(len(index_ax1)):
	pathDicom_ax.append(pathDicom+index_ax1[i]+"/ax_"+index_ax2[i]+"/")
	pathNifti_ax.append(pathNifti+index_ax1[i]+"_ax_"+index_ax2[i]+"_stanford_nml_seg.nii.gz")
	nifti = nib.load(pathNifti_ax[i])
	nifti_shape = nifti.get_fdata().shape
	lstFilesDCM = []  # create an empty list
	for dirName, subdirList, fileList in os.walk(pathDicom_ax[i]):
		for filename in fileList:
			lstFilesDCM.append(os.path.join(dirName,filename))
	if len(lstFilesDCM) != nifti_shape[2]:
		f.write("Error of ax: index "+str(i)+"; dicom directory is "+pathDicom_ax[i]+"; nifti files is "+pathNifti_ax[i]+".\n")
	else:
		ds = pydicom.read_file(lstFilesDCM[0])
		if ds.pixel_array.shape[0] != nifti_shape[0]:
			f.write("Error of ax: index "+str(i)+"; nifti files is "+pathNifti_ax[i]+"; nifti dimension is "+str(nifti.shape[0])+" and dicom dimension is "+str(ds.pixel_array[0]))
		#print(str(nifti_shape[0]))

for i in range(len(index_cor1)):
	pathDicom_cor.append(pathDicom+index_cor1[i]+"/cor_"+index_cor2[i]+"/")
	pathNifti_cor.append(pathNifti+index_cor1[i]+"_cor_"+index_cor2[i]+"_stanford_nml_seg.nii.gz")
	nifti = nib.load(pathNifti_cor[i])
	nifti_shape = nifti.get_fdata().shape
	lstFilesDCM = []  # create an empty list
	for dirName, subdirList, fileList in os.walk(pathDicom_cor[i]):
		for filename in fileList:
			lstFilesDCM.append(os.path.join(dirName,filename))
	if len(lstFilesDCM) != nifti_shape[2]:
		f.write("Error of cor: index "+str(i)+"; dicom directory is "+pathDicom_cor[i]+"; nifti files is "+pathNifti_cor[i]+".\n")
	else:
		ds = pydicom.read_file(lstFilesDCM[0])
		if ds.pixel_array.shape[0] != nifti_shape[0]:
			f.write("Error of cor: index "+str(i)+"; nifti files is "+pathNifti_cor[i]+"; nifti dimension is "+str(nifti.shape[0])+" and dicom dimension is "+str(ds.pixel_array[0]))
		#print(str(nifti_shape[0]))

for i in range(len(index_sag1)):
	pathDicom_sag.append(pathDicom+index_sag1[i]+"/sag_"+index_sag2[i]+"/")
	pathNifti_sag.append(pathNifti+index_sag1[i]+"_sag_"+index_sag2[i]+"_stanford_nml_seg.nii.gz")
	nifti = nib.load(pathNifti_sag[i])
	nifti_shape = nifti.get_fdata().shape
	lstFilesDCM = []  # create an empty list
	for dirName, subdirList, fileList in os.walk(pathDicom_sag[i]):
		for filename in fileList:
			lstFilesDCM.append(os.path.join(dirName,filename))
	if len(lstFilesDCM) != nifti_shape[2]:
		f.write("Error of sag: index "+str(i)+"; dicom directory is "+pathDicom_sag[i]+"; nifti files is "+pathNifti_sag[i]+".\n")
	else:
		ds = pydicom.read_file(lstFilesDCM[0])
		if ds.pixel_array.shape[0] != nifti_shape[0]:
			f.write("Error of sag: index "+str(i)+"; nifti files is "+pathNifti_sag[i]+"; nifti dimension is "+str(nifti.shape[0])+" and dicom dimension is "+str(ds.pixel_array[0]))
		#print(str(nifti_shape[0]))
f.close()
