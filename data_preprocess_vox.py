# -*- coding: utf-8 -*-
"""
Coppyright 2018 Zhonghao Guo

guozhonghao1994@live.cn

2018.7

"""
import os
import librosa
from configuration import get_config
import numpy as np

config = get_config()

os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file(Python3.x only)
os.makedirs(config.test_path, exist_ok=True)  # make folder to save test file
#os.makedirs(config.verif_path, exist_ok=True)   # make folder to save verif file

audio_path = r'C:\Users\guozh\Documents\intern\minivox'
text_path = r'C:\Users\guozh\Documents\intern\GE2E_vox'
text_name = 'voxceleb1_test.txt'

# part 1---test speaker and training speaker list
test_speaker = []
for line in open(os.path.join(text_path,text_name)):
    test_speaker.append(line.split()[1].split("/")[0])
    test_speaker.append(line.split()[2].split("/")[0])

test_speaker = list(set(test_speaker))
train_speaker = []
train_speaker = list(set(os.listdir(audio_path))^set(test_speaker))
#print(train_speaker)
#print(c)

# part 2--create enrollment and verification list
enroll_utter_list = []
verif_utter_list = []
test_utter_list = []
for num in open(os.path.join(text_path,text_name)):
    enroll_utter_list.append(num.split()[1].split("/")[1])
    verif_utter_list.append(num.split()[2].split("/")[1])
test_utter_list = list(set(enroll_utter_list + verif_utter_list))
enroll_utter_list = list(set(enroll_utter_list))
verif_utter_list = list(set(verif_utter_list))
test_utter_list = list(set(test_utter_list))

utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr  # 5320


# part3---generate testing utterances' numpy files(one 'wav',one numpy)
for i, folder in enumerate(test_speaker):
    speaker_path = os.path.join(audio_path, folder)     # path of each speaker
    #print(speaker_path)
    print("%dth testing speaker processing..."%i)
    print("speaker name: {}".format(folder))
    utterances_spec = []
    
    for utter_name in os.listdir(speaker_path):
        if utter_name in test_utter_list:
            #print(utter_name)
            utter_path = os.path.join(speaker_path, utter_name)
            utter, sr = librosa.core.load(utter_path, config.sr)
            intervals = librosa.effects.split(utter, top_db=20)
            S_utter = np.ndarray(shape=(40,0))
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:
                    utter_part = utter[interval[0]:interval[1]]
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                    # Combine all intervals
                    S_utter = np.concatenate((S_utter,S),axis=1)
            """
            # judge if S_utter is empty. If so, break the loop
            if S_utter.shape[1] == 0:
                break
            """
            #S_utter.tolist()   # Change format to list
            
            S_moving_sum = []
            for step_num in range(S_utter.shape[1]-config.tisv_frame):
                #print(S_utter[:,step_num:step_num + config.tisv_frame].shape)
                S_moving_sum.append(S_utter[:,step_num:step_num + config.tisv_frame])
            S_utter = sum(S_moving_sum)/(S_utter.shape[1]-config.tisv_frame)
            #print(S_utter.shape)
            np.save(os.path.join(config.test_path,"{}.npy".format(utter_name[:-4])),S_utter)

           
# part4---generate training utterances' numpy files(one speaker,one numpy)
for i, folder in enumerate(train_speaker):     
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth training speaker processing..."%i)
        print("speaker name: {}".format(folder))
        existing_npy_list = []
        for exist in os.listdir(config.train_path):         # resume from where it crushed last time
            existing_npy_list.append(exist[:-4])
        if folder in existing_npy_list:
            print("numpy file exsited")
            continue
        utterances_spec = []            # initialize the matrix that will be saved into numpy file
        
        for utter_name in os.listdir(speaker_path):
            #print(utter_name)
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance（这里列出了文件夹下【每一个说话人】所有的声音文件）
            utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio（加载每个说话人的全部语音）
            intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection（检测每个语音的活动区间）
            S_utter = np.ndarray(shape=(40,0))
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                    S_utter = np.concatenate((S_utter,S),axis=1)        # Combine all intervals together 
            # If S_utter is empty(voiced intervals are all too short), skip this utter
            if S_utter.shape[1] == 0:
                break
            #S_utter.tolist()   # Change format to list
            S_moving_sum = []
            #print(S_utter.shape)
            for step_num in range(S_utter.shape[1]-config.tisv_frame):
                #print(S_utter[:,step_num:step_num + config.tisv_frame].shape)
                S_moving_sum.append(S_utter[:,step_num:step_num + config.tisv_frame])
            S_utter = sum(S_moving_sum)/(S_utter.shape[1]-config.tisv_frame)
            
            #print(S_utter.shape)
            utterances_spec.append(S_utter[:,:])       
            #print(utterances_spec)
        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        # save spectrogram as numpy file in train folder
        np.save(os.path.join(config.train_path, "{}.npy".format(folder)),utterances_spec)

     
print("==========================")             
print("preprocess done!")   
  