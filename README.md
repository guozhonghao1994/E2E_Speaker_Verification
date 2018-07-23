# TDSV_nihaoxiaogua
This branch is mostly done by using our own dataset --- nihaoxiaogua

### What is nihaoxiaogua?
Nihaoxiaogua is a self-made dataset by Lenovo(Beijing) AI Lab. Totally we have 113 speakers, who has 40 utterances each. To expand the dataset, we add different level of noise(0dB/5dB/10dB/15dB/20dB) onto each audio file. 101 of which are used to train neural network and the other 12 are for testing.

### What's the uniqueness?
- utils.py
  The original batch size, 64x10, seems to be too large to be utilized here. Hence, we minimize it to be 5x10, that is, randomly choose 5 spks and 10 utters of whom at a time. 

- models.py
  For each spk, we have totally 40x6=240 utters. In testing, we make a full combination of those 240 utters. Thus we have 300k+ pairs of "match" case. To make our data balanced, another 30k+ "unmatch" pairs are randomly chosen in rest of data. 
  
  All cos similarities are saved in cos_similarity.txt, which is located in model_path.

- ROC.py
  For those servers without visulized tools, you might simply download the similarity text file to local PC and calculate FAR,FRR,EER and plot your own ROC curve.
  
- data_preprocess_nihao.py
  The original data is saved in pickle format on server. The thing we should do first is to transform to numpy format.
  
### Result
The whole experiment was done on a platform of 2 E5-2699 v4 CPUs and 4 Tesla P100 GPUs.

Matrix loss was about 10 after 2m times iterations.

Pair by pair of testing was time consuming!

The EER is xxx

ROC curve and tensorboard curve:
  

