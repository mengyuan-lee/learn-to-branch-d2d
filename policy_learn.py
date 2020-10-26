# -*- coding:utf-8 -*-
#-----------------------------------------------------------
# training and testing
# imitation learning
#-----------------------------------------------------------
import scipy.io as sio  
import numpy as np
from svmutil import *
import bb_oracle_dele as bo
import bb_policy_dele as bp
import policy_run as pr
from sklearn import preprocessing
import re
import os


def prune_learn(matfn,matva,Nround,Nprob,Nval):
	# Nround：iteration rounds
	# Nprob：the number of instances
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

	data1= sio.loadmat(matva)
	K1 = data1['K'][0,0]
	L1 = data1['L'][0,0]
	R_min_C1 = data1['R_min_C'][0,0]
	P_max_D1 = data1['P_max_D'][0,0]
	P_max_C1 = data1['P_max_C'][0,0]
	H_CD1 = np.transpose(data1['H_CD'])
	H_D1 = np.transpose(data1['H_D'])
	H_CB1 = np.transpose(data1['H_CB'])
	H_DB1 = np.transpose(data1['H_DB'])
	 
	# generate training data
	train_path = 'data_'+str(K)+'_'+str(L)+'/train.txt'
	if os.path.exists(train_path):
		os.remove(train_path)

	# generate weight file
	wePath = 'data_'+str(K)+'_'+str(L)+'/weight.txt'
	if os.path.exists(wePath):
		os.remove(wePath)

	# generate file to record accuracy
	gap_path = 'data_'+str(K)+'_'+str(L)+'/gap.txt'

	# path to store svm model
	dirs = 'data_'+str(K)+'_'+str(L)+'/model'
	if not os.path.exists(dirs):
		os.makedirs(dirs)


	# imitation learning
	for r in range(Nround):
		print("round #%d..."%(r+1))
		for i in range(Nprob):
			print("problem #%d..."%(i+1))
			h_CD = H_CD[:,i].reshape((K,L))
			h_D = H_D[:,i]
			h_CB = H_CB[:, i]
			h_DB = H_DB[:, i]
			index = i + 1
			if r==0:
				#u sing oracle prune policy 
				label_train_tmp,feature_train_tmp = bo.binaryPro_oracle(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,index)
			else:
				# using prune policy learned with last iteration
				label_train_tmp,feature_train_tmp = bp.binaryPro_policy(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,index,m,scaler)
			if r == 0 and i == 0:
				label_train  = label_train_tmp
				feature_train = feature_train_tmp
			else:
				label_train  = np.hstack((label_train,label_train_tmp))
				feature_train = np.vstack((feature_train,feature_train_tmp))
		weFile = open(wePath, mode='r')
		sourceInLine=weFile.readlines()
		W=[]
		W_temp = []
		for line in sourceInLine:
			temp=line.strip('\n')
			W_temp.append(temp)
		W = list(map(eval,W_temp))
		weFile.close()


		feature_scaled = preprocessing.scale(feature_train)
		scaler = preprocessing.StandardScaler().fit(feature_train)

		train_file = open(train_path,mode='w+')
		train_num = feature_scaled.shape[0]
		for i in range(train_num):
			train_file.writelines([str(label_train[i])])
			for j in range(len(feature_train[i,:])):
				train_file.writelines(['\t',str(j),':' , str(feature_scaled[i,j])])
			train_file.writelines(['\n'])
		train_file.close()

		num1 = 0
		num2 = 0
		train_file = open(train_path,mode='r')
		line = train_file.readline()
		while line != "":
			s = line[:1]
			if s == str(1):
				num1 = num1 + 1
			else:
				num2 = num2 + 1
			line = train_file.readline()
		train_file.close()       
		
		y_, x = svm_read_problem(train_path)
		m = svm_train(W, y_, x, '-c 0.25 -w1 8')
		print('test for round %d:'%(r+1))

		svm_save_model('data_'+str(K)+'_'+str(L)+'/model/model_'+ str(r+1)+'.model', m)
		gap_file = open(gap_path,mode='a+')
		gap_file.writelines(['round:\t',str(r+1),'\n'])
		gap_file.close()
		gap_sum = 0
		speed_sum = 0
		acc1_sum = 0
		acc0_sum = 0
		num1_sum = 0
		num0_sum = 0


		# test learned model
		for i in range(0,Nval):
			print("problem #%d..."%(i+1))
			h_CD1 = H_CD1[:,i].reshape((K,L))
			h_D1 = H_D1[:,i]
			h_CB1 = H_CB1[:, i]
			h_DB1 = H_DB1[:, i]
			index = i + 1
			ogap, speed, acc1,acc0,num1,num0 = pr.policyRun(K1,L1,R_min_C1,P_max_D1,P_max_C1,h_CD1,h_D1,h_CB1,h_DB1,index,m,scaler)
			gap_file = open(gap_path,mode='a+')
			gap_file.writelines(['problem:\t',str(index), '\tOgap:\t', str(ogap),'\n'])
			gap_file.close()
			gap_sum = gap_sum + ogap
			speed_sum = speed_sum + speed
			acc1_sum = acc1_sum + acc1
			acc0_sum = acc0_sum + acc0
			num1_sum = num1_sum + num1
			num0_sum = num0_sum + num0
		gap_file = open(gap_path,mode='a+')
		gap_file.writelines(['average gap:\t',str(gap_sum/Nval),'\taverage speed:\t',str(speed_sum/Nval),'\tacc1:\t',str(acc1_sum/num1_sum),'\tacc0:\t',str(acc0_sum/num0_sum), '\n'])
		gap_file.close()





