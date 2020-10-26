# -*- coding:utf-8 -*-
#-----------------------------------------------------------
# generate .txt file for data
#-----------------------------------------------------------
import scipy.io as sio  
import numpy as np
import bb 
import os

def data_generate(matfn,mode,num_total):
	data= sio.loadmat(matfn)
	K = data['K'][0,0]
	L = data['L'][0,0]
	R_min_C = data['R_min_C'][0,0]
	P_max_D = data['P_max_D'][0,0]
	P_max_C = data['P_max_C'][0,0]
	H_CD = np.transpose(data['H_CD'])
	H_D = np.transpose(data['H_D'])
	H_CB = np.transpose(data['H_CB'])
	H_DB = np.transpose(data['H_DB'])


	
	for i in range(0,num_total):
		print("solving problem #%d..."%(i+1))
		h_CD = H_CD[:,i].reshape((K,L))
		h_D = H_D[:,i]
		h_CB = H_CB[:, i]
		h_DB = H_DB[:, i]
		# using b&b to solve the i-th instance and store the results
		index = i+1
		rho, yita_max, label_temp, feature_temp = bb.binaryPro(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,index,mode)
		if mode == 1 and len(label_temp) != 0:
			if i == 0:
				label = label_temp
				feature = feature_temp
			else:
				label = np.vstack((label,label_temp))
				feature = np.vstack((feature,feature_temp))
		else:
			label = np.array([])
			feature = np.array([])

		if mode == 1:
			labelPath = 'data_'+str(K)+'_'+str(L)+'/val/label_val.txt'
			featurePath = 'data_'+str(K)+'_'+str(L)+'/val/feature_val.txt'

			if os.path.exists(labelPath):
				os.remove(labelPath)
			if os.path.exists(featurePath):
				os.remove(featurePath)
			np.savetxt(featurePath,feature)
			np.savetxt(labelPath,label)

		print("problem #%d solved!"%(i+1))
		print("final solution:",rho)
		print("optimal value:",yita_max)
	return label, feature
