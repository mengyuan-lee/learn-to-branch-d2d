# -*- coding:utf-8 -*-
#--------------------------------------------------------
# main
#--------------------------------------------------------

import math
import sys
import numpy as np
import matlab
import matlab.engine
import os
from svmutil import *
import data_generate as dg
import policy_learn as pl

eng = matlab.engine.start_matlab()

def main(argv=None):
	num_train = 50 #t he number of training samples
	num_val = 20 # the number of testing samples
	Nround = 3

	K = 5
	L = 3


	eng.main_generate(K,L,num_train,'train',nargout=0) # generate training data
	mattr='data_'+str(K)+'_'+str(L)+'_train.mat'  # path of training data
	dirs = 'data_'+str(K)+'_'+str(L)+'/oracle'
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	print("generate oracle...")
	dg.data_generate(mattr,0,num_train)

	eng.main_generate(K,L,num_train,'val',nargout=0) # generate testing data
	matva='data_'+str(K)+'_'+str(L)+'_val.mat'  # path of testing data
	dirs = 'data_'+str(K)+'_'+str(L)+'/val'
	if not os.path.exists(dirs):
		os.makedirs(dirs)

	print("generate validation set...")
	label,feature = dg.data_generate(matva,1,num_val)
	print('learn and validate policy...')
	pl.prune_learn(mattr,matva,Nround,num_train,num_val)



if __name__ == "__main__":
	main()