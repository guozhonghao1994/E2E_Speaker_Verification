# TISV_voxceleb1
Implementation of generalized end-to-end loss for speaker verification on voxceleb1 dataset.


### What is voxceleb1 and where can I get it?
https://arxiv.org/abs/1706.08612

http://www.robots.ox.ac.uk/~vgg/data/voxceleb/


### Tips
- data_preprocess.py  
  voice activity detection is performed by using librosa library.

  A 64-length window moves from the beginning to the end on each utterance, giving the moving average of a utter, generating a 64-dim ndarray then(d-vector).

  Each test utter will be transformed to a npy file.

- utils.py   
  random_batch and test_batch are modified. Considering our limited calculating resources, M & N are slightly minimized. 

  When evaluating the similarity between enroll utter and verification utter, cosine similarity is used, not in matrix form.

- model.py  
  Testcase is shown in voxceleb1_test.txt. Test is done by calculating the similarity of 30k+ pairs, with match/unmatch evenly distributed.


### Results
  I trained the model with 8 P100 GPU. Model hyperpameters are followed by the paper :3-lstm layers with 768 hidden nodes and 256 projection nodes, 0.01 lr sgd with 1/2 decay, l2 norm clipping with 3.

  When we fixed M * N to 64x10, the "matrix loss" maintained at a very high level, approximately 1300 or more and stucked there even after 1m iterations. 

  No further test was done yet.










