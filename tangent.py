import numpy as np
from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import math
import os
import time

labels_path = "Data/Phenotypic_data.csv"
df = pd.read_csv(labels_path)

Data = []
Data_VIQ = []
Data_FIQ = []
Data_PIQ = []

for i in range(df.shape[0]):
	path = "Data/Outputs/ccs/nofilt_noglobal/rois_cc200/" + df.iloc[i]["FILE_ID"] + "_rois_cc200.1D"
	viq = float(df.iloc[i]["VIQ"])
	piq = float(df.iloc[i]["PIQ"])
	fiq = float(df.iloc[i]["FIQ"])


	if os.path.isfile(path):

		viq_flag = False
		piq_flag = False
		fiq_flag = False

		if not (math.isnan(viq) or viq<0):
			Data_VIQ.append([path,viq])
			viq_flag = True

		if not (math.isnan(piq) or piq<0):
			Data_PIQ.append([path,piq])
			piq_flag = True

		if not (math.isnan(fiq) or fiq<0):
			Data_FIQ.append([path,fiq])
			fiq_flag = True

		if viq_flag and piq_flag and fiq_flag:
			Data.append([path,viq,piq,fiq])


print(len(Data))
print(len(Data_VIQ))
print(len(Data_PIQ))
print(len(Data_FIQ))




roi_time_series_viq = []

start = time.time()

for i in range(len(Data_VIQ)):

	print(i)

	path = Data_VIQ[i][0]
	file = open(path,'r')
	L = file.readlines()
	file.close()

	data = []

	for i in range(1,len(L)):

		roi = L[i].split()
		y = [float(x) for x in roi]
		data.append(y)

	data = np.array(data)
	roi_time_series_viq.append(data)

tangent_measure = ConnectivityMeasure(kind='tangent')
tangent_matrices = tangent_measure.fit_transform(roi_time_series_viq)

end = time.time()
print(end-start)

print(tangent_matrices.shape)
print(tangent_matrices.dtype)
np.save("Data/no_filt_tangent_cc200_VIQ.npy", tangent_matrices)




labels_viq = np.array([lab[1] for lab in Data_VIQ])
np.save("Data/labels_cc200_VIQ.npy", labels_viq)






roi_time_series_piq = []

start = time.time()

for i in range(len(Data_PIQ)):

	print(i)

	path = Data_PIQ[i][0]
	file = open(path,'r')
	L = file.readlines()
	file.close()

	data = []

	for i in range(1,len(L)):

		roi = L[i].split()
		y = [float(x) for x in roi]
		data.append(y)

	data = np.array(data)
	roi_time_series_piq.append(data)

tangent_measure = ConnectivityMeasure(kind='tangent')
tangent_matrices = tangent_measure.fit_transform(roi_time_series_piq)

end = time.time()
print(end-start)

print(tangent_matrices.shape)
print(tangent_matrices.dtype)
np.save("Data/no_filt_tangent_cc200_PIQ.npy", tangent_matrices)


labels_piq = np.array([lab[1] for lab in Data_PIQ])
np.save("Data/labels_cc200_PIQ.npy", labels_piq)








roi_time_series_fiq = []

start = time.time()

for i in range(len(Data_FIQ)):

	print(i)

	path = Data_FIQ[i][0]
	file = open(path,'r')
	L = file.readlines()
	file.close()

	data = []

	for i in range(1,len(L)):

		roi = L[i].split()
		y = [float(x) for x in roi]
		data.append(y)

	data = np.array(data)
	roi_time_series_fiq.append(data)

tangent_measure = ConnectivityMeasure(kind='tangent')
tangent_matrices = tangent_measure.fit_transform(roi_time_series_fiq)

end = time.time()
print(end-start)

print(tangent_matrices.shape)
print(tangent_matrices.dtype)
np.save("Data/no_filt_tangent_cc200_FIQ.npy", tangent_matrices)



labels_fiq = np.array([lab[1] for lab in Data_FIQ])
np.save("Data/labels_cc200_FIQ.npy", labels_fiq)