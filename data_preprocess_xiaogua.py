# -*- coding: utf-8 -*-
"""
Coppyright 2018 Zhonghao Guo

guozhonghao1994@live.cn

2018.7

"""
import os
import numpy as np
import pickle
from configuration import get_config


config = get_config()

os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
os.makedirs(config.test_path, exist_ok=True)  # make folder to save test file

txt_path = r'C:\Users\guozh\Documents\intern\GE2E_nihao'

# part1---training
""" Read pickles file from given txt. Then calculate the moving average
using a 64-length window. Trnasform to numpy file.
"""

# txt files for clean dataset and noised dataset
txt_name = ['train_nihao_with_sil_spk.txt',
			'train_nihao_with_sil_snr0_spk.txt',
			'train_nihao_with_sil_snr5_spk.txt',
			'train_nihao_with_sil_snr10_spk.txt',
			'train_nihao_with_sil_snr15_spk.txt',
			'train_nihao_with_sil_snr20_spk.txt']

total_spk_num = 113		# total speaker num, 101 of which are training spk
for spk_num in range(1,total_spk_num+1):
	utterances_spec = []		# the utterances_spec of spk_num(th) speaker
	for txt_num in range(len(txt_name)):
		for pickle_path in open(os.path.join(txt_path,txt_name[txt_num])):
			# the pickle path in the txt file
			with open(pickle_path[:-1],'rb') as f:
				data = pickle.load(f,encoding='iso-8859-1')
				s = data['spks']
				if s[0] == spk_num:
					f = data['featMat']
					S_moving_sum = []
					for step_num in range(f.shape[0]-config.tisv_frame):
						S_moving_sum.append(f[step_num:step_num + config.tisv_frame,:])
					S_utter = sum(S_moving_sum)/(f.shape[0]-config.tisv_frame)
					S_utter = np.transpose(S_utter)
					utterances_spec.append(S_utter[:,:])
	if utterances_spec:
		utterances_spec = np.array(utterances_spec)
	else:
		continue

	print("speaker num: {}".format(spk_num))
	print(utterances_spec.shape)
	np.save(os.path.join(config.train_path, "{}.npy".format(spk_num)),utterances_spec)
	print("------------------")
print("training data extraction done")

# part2---testing
""" Read pickles files from testing txt. Then calculate the moving average
using a 64-length window. Trnasform to numpy file.
"""	
test_txt_name = ['eval_nihao_with_sil_snr0_spk.txt',
				'eval_nihao_with_sil_snr5_spk.txt',
				'eval_nihao_with_sil_snr10_spk.txt',
				'eval_nihao_with_sil_snr15_spk.txt',
				'eval_nihao_with_sil_snr20_spk.txt',
				'eval_nihao_with_sil_spk.txt',
				'test_nihao_with_sil_snr0_spk.txt',
				'test_nihao_with_sil_snr5_spk.txt',
				'test_nihao_with_sil_snr10_spk.txt',
				'test_nihao_with_sil_snr15_spk.txt',
				'test_nihao_with_sil_snr20_spk.txt'
				'test_nihao_with_sil_spk.txt']

for txt_num in range(len(test_txt_name)):
	for i, pickle_path in enumerate(open(os.path.join(txt_path,test_txt_name[txt_num]))):
		utterances_spec = []
		with open(pickle_path[:-1],'rb') as f:
			data = pickle.load(f,encoding='iso-8859-1')
			s = data['spks']
			test_spk_num = s[0]
			f = data['featMat']
			S_moving_sum = []
			for step_num in range(f.shape[0]-config.tisv_frame):
				S_moving_sum.append(f[step_num:step_num + config.tisv_frame,:])
			S_utter = sum(S_moving_sum)/(f.shape[0]-config.tisv_frame)
			S_utter = np.transpose(S_utter)
			utterances_spec.append(S_utter[:,:])
			utterances_spec = np.array(utterances_spec)
			#print(utterances_spec.shape)
			
			np.save(os.path.join(config.test_path, "{}_{}_{}.npy".format(test_spk_num,txt_num+1,i+1)),utterances_spec)

print("testing data extraction done")
print("all done")