import tensorflow as tf
import numpy as np
import os
import itertools
from utils import random_batch, normalize, similarity, loss_cal, optim, cossim, test_batch
from configuration import get_config
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import shapely.geometry as SG

config = get_config()
#text_path = r'C:\Users\guozh\Documents\intern\GE2E_nihao'
#text_name = 'voxceleb1_test.txt'

def train(path):
    tf.reset_default_graph()    # reset graph

    # draw graph
    batch = tf.placeholder(shape= [None, config.N*config.M, 40], dtype=tf.float32)  # input batch (time x batch x n_mel)
    lr = tf.placeholder(dtype= tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer= np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float32))

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    print("embedded size: ", embedded.shape)

    # loss
    sim_matrix = similarity(embedded, w, b)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, type=config.loss)

    # optimizer operation
    trainable_vars= tf.trainable_variables()                # get variable list
    optimizer = optim(lr)                                   # get optimizer (type is determined by configuration)
    grads, vars= zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss
    grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)      # l2 norm clipping by 3
    grads_rescale= [0.01*grad for grad in grads_clip[:2]] + grads_clip[2:]   # smaller gradient scale for w, b
    train_op= optimizer.apply_gradients(zip(grads_rescale, vars), global_step= global_step)   # gradient update operation

    # check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # record loss
    loss_summary = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    # training session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        os.makedirs(os.path.join(path, "Check_Point"), exist_ok=True)  # make folder to save model
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)         # make folder to save log
        writer = tf.summary.FileWriter(os.path.join(path, "logs"), sess.graph)
        #epoch = 0
        lr_factor = 1   # lr decay factor ( 1/2 per 10000 iteration)
        loss_acc = 0    # accumulated loss ( for running average of loss)

        for iter in range(config.iteration):
            # run forward and backward propagation and update parameters
            _, loss_cur, summary = sess.run([train_op, loss, merged],
                                  feed_dict={batch: random_batch(), lr: config.lr*lr_factor})
            print("iteration {} is processing, loss: {}".format(iter,loss_cur))
            loss_acc += loss_cur    # accumulated loss for each 100 iteration

            if iter % 10 == 0:
                writer.add_summary(summary, iter)   # write at tensorboard
            if (iter+1) % 100 == 0:
                print("(iter : %d) loss: %.4f" % ((iter+1),loss_acc/100))
                loss_acc = 0                        # reset accumulated loss
            if (iter+1) % 10000 == 0:
                lr_factor /= 2                      # lr decay
                print("learning rate is decayed! current lr : ", config.lr*lr_factor)
            if (iter+1) % 10000 == 0:
                saver.save(sess, os.path.join(path, "./Check_Point/model.ckpt"), global_step=iter//10000)
                print("model is saved!")


# Test Session
def test(path):
    
    tf.reset_default_graph()
    # draw graph
    enroll = tf.placeholder(shape=[None, 1, 40], dtype=tf.float32) # enrollment batch (time x 1*1 x n_mel)
    verif = tf.placeholder(shape=[None, 1, 40], dtype=tf.float32)  # verification batch (time x 1*1 x n_mel)
    batch = tf.concat([enroll, verif], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # enrollment embedded vectors (speaker model)
    #enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:64, :], shape= [1, 1, -1]), axis=1))
    enroll_embed = embedded[:1,:]
    # verification embedded vectors
    verif_embed = embedded[1:, :]
    
    
    cos_similarity = cossim(enroll_embed,verif_embed,normalized=True)
    #similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # load model
        print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        print("test file path : ", config.test_path)

        # read test files pair by pair and get all combinations
        utter_name_list =[]
        for utter_name in os.listdir(config.test_path):
            utter_name_list.append(utter_name)
        comb = []       # combinations of all pairs
        for i in itertools.combinations(utter_name_list,2):
            if i[0].split("_")[0] == i[1].split("_")[0]:
                comb.append(i)
            
        match_length = len(comb)
        random.shuffle(utter_name_list)
        for j in itertools.combinations(utter_name_list,2):
            if j[0].split("_")[0] != j[1].split("_")[0] and len(comb)<2*match_length:
                comb.append(j)

        f = open("./tisv_model/combinations.txt",'w')
        for i in range(len(comb)):
            f.write(str(comb[i])+'\n')     # column 1: cos sim column 2: label
        f.close()
        print("combinations saved!")
        
        # calculate all cos similarities
        Sim = []
        Label = []
        for pair_num in open("./tisv_model/combinations.txt"):
            if pair_num[0].split("_")[0] == pair_num[1].split("_")[0]:
                label = 'match'     # if spk_num1 == spk_num2, label = match
            else:
                label = 'unmatch'   # if spk_num1 != spk_num2, label = unmatch
            enroll_utter_name = pair_num[0]
            verif_utter_name = pair_num[1]

            S = sess.run(cos_similarity, feed_dict={enroll:test_batch(enroll_utter_name),
                                                    verif:test_batch(verif_utter_name)})
            Sim.append(S)       # calculate cos similarity of each pair and append them into Sim
            Label.append(label)
        # save Sim as a text file in model_path
        f = open("./tisv_model/cos_similarity.txt",'w')
        for i in range(len(Sim)):
            f.write(str(Sim[i])+" "+str(Label[i])+'\n')     # column 1: cos sim column 2: label
        f.close()

        # calculate FAR,FRR and EER
        FAR_list = []; FRR_list = []; total_match = 0; total_unmatch = 0; FA = 0; FR = 0
        # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
        for thres in [0.01*i-1.0 for i in range(200)]:      # Generate [-1,1] with interval 0.01
            for pair_num, pair in open("./tisv_model/cos_similarity.txt"):  # load evaluation pairs 
                flag = pair.split()[1]  # label of pair
                if flag == 'unmatch':
                    total_unmatch += 1
                    if Sim[pair_num] > thres:
                        FA += 1
                elif flag == 'match':
                    total_match +=1
                    if Sim[pair_num] < thres:
                        FR += 1
            FAR = FA/total_unmatch      # the FAR under current thres
            FAR_list.append(FAR)
            FRR = FR/total_match        # the FRR under current thres
            FRR_list.append(FRR)

        plt.plot(FAR_list,FRR_list)     # plot ROC
        plt.show()

        # get estimated EER by calculating relative gap between FAR and FRR 
        gap = []
        for i in range(len(FAR_list)):
            gap.append(abs(FAR_list[i]-FRR_list[i]))
        #print(gap)
        min_pos = gap.index(min(gap))
        #print(min_pos)
        print("EER: {}".format((FAR_list[min_pos]+FRR_list[min_pos])/2))