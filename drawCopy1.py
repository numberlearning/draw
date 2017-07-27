#!/usr/bin/env python

""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow

Example Usage: 
	python draw.py --data_dir=/tmp/draw --read_attn=True --write_attn=True

Author: Eric Jang
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import time
import sys
import load_trace

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 

A,B = 100,100 # image width,height
img_size = B*A # the canvas size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
read_n = 10#12 # read glimpse grid width/height
write_n = 5#12 # write glimpse grid width/height
read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n if FLAGS.write_attn else img_size
z_size = 9#10#2 # QSampler output size
T = 5#100 # MNIST generation sequence length
batch_size = 1#00 # training minibatch size
train_iters = 500000
learning_rate = 1e-3 # learning rate for optimizer
eps = 1e-8 # epsilon for numerical stability

## BUILD MODEL ## 

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, z_size))
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True) # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy


def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params,5,1)
    gx1=(A+1)/2*(gx_+1)
    gy1=(B+1)/2*(gy_+1)

    gx = gx1
    gy = gy1
#    gx = tf.where(tf.less(gx1, tf.zeros_like(gx1) + A), gx1, tf.zeros_like(gx1) + A)
#    gx = tf.where(tf.greater(gx1, tf.zeros_like(gx1)), gx1, tf.zeros_like(gx1))
#    gy = tf.where(tf.less(gy1, tf.zeros_like(gy1) + B), gy1, tf.zeros_like(gy1) + B)
#    gy = tf.where(tf.greater(gy1, tf.zeros_like(gy1)), gy1, tf.zeros_like(gy1))

    sigma2=tf.exp(log_sigma2)
    d = (max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N

    delta = d
#    delta = tf.where(tf.less(d, tf.zeros_like(d) + A / read_n), d, tf.zeros_like(d) + A / read_n)

    Fx, Fy = filterbank(gx,gy,sigma2,delta,N) 
    gamma = tf.exp(log_gamma)
    return Fx, Fy, gamma, gx, gy, delta

## READ ## 
def read(x,h_dec_prev):
    Fx,Fy,gamma, gx, gy, delta=attn_window("read",h_dec_prev,read_n)
    stats = Fx, Fy, gamma
    new_stats = gx, gy, delta 

    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])

    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    return x, new_stats

## ENCODE ## 
def encode(input, state):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder/LSTMCell",reuse=DO_SHARE):
        return lstm_enc(input,state)

## DECODER ## 
def decode(input, state):
    with tf.variable_scope("decoder/LSTMCell",reuse=DO_SHARE):
        return lstm_dec(input, state)

## STATE VARIABLES ## 
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ## 

viz_data = list()
pqs = list()

# construct the unrolled computational graph
for t in range(T):
    r, stats = read(x, h_dec_prev)
   
    h_enc, enc_state = encode(tf.concat([r, h_dec_prev], 1), enc_state)

    with tf.variable_scope("z",reuse=DO_SHARE):
        z = linear(h_enc, z_size)
    h_dec, dec_state = decode(z, dec_state)
    h_dec_prev = h_dec

    with tf.variable_scope("hidden1",reuse=DO_SHARE):
        hidden = tf.nn.relu(linear(h_dec_prev, 256))
    with tf.variable_scope("output",reuse=DO_SHARE):
        classification = tf.nn.softmax(linear(hidden, z_size))
        viz_data.append({
            "classification": classification,
            "r": r,
            "h_dec": h_dec,
            "stats": stats,
        })

    DO_SHARE=True # from now on, share variables

    pq = tf.log(classification + 1e-5) * onehot_labels
    pq = tf.reduce_mean(pq, 0)
    pqs.append(pq)

predquality = tf.reduce_mean(pqs)
correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)

R = tf.cast(tf.equal(correct, prediction), tf.float32)

reward = tf.reduce_mean(R)


## LOSS FUNCTION ## 

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))


def evaluate():
    data = load_trace.TraceData()
    data.get_test(1)
    batches_in_epoch = len(data.images) // batch_size
    accuracy = 0
    
    for i in range(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        feed_dict = {x: nextX, onehot_labels:nextY}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy


predcost = -predquality


## OPTIMIZER ## 

optimizer=tf.train.AdamOptimizer(learning_rate, epsilon=1)
grads=optimizer.compute_gradients(predcost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

## RUN TRAINING ## 

#data_directory = os.path.join(FLAGS.data_dir, "mnist")
#if not os.path.exists(data_directory):
#    os.makedirs(data_directory)
#train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data

train_data = load_trace.TraceData()
train_data.get_train()

fetches=[]
fetches.extend([reward, train_op])

if __name__ == '__main__':
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess=tf.InteractiveSession()

    saver = tf.train.Saver() # saves variables learned during training
    tf.global_variables_initializer().run()

    ## CHANGE THE MODEL SETTINGS HERE #########################
    model_directory = "model_runs/blob_classification_5_5_0_9"

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    start_ckpt = 0 
    #saver.restore(sess, model_directory + "/drawmodel.ckpt") # to restore from model, uncomment this line
    #saver.restore(sess, model_directory + "/drawmodel_" + str(start_ckpt) + ".ckpt") # to restore from model, uncomment this line, may need to change filename!!!

    start_time = time.clock()
    extra_time = 0

    for i in range(start_ckpt, train_iters):
        xtrain, ytrain = train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
        feed_dict={x:xtrain, onehot_labels: ytrain}
        results=sess.run(fetches,feed_dict)
        reward_fetched, _ = results
        if i%100 == 0:
            print("iter=%d : Reward: %f" % (i, reward_fetched))
            sys.stdout.flush()

            if i%1000==0:
                train_data = load_trace.TraceData()
                train_data.get_train()
     
                if i %10000==0:
                    ## SAVE TRAINING CHECKPOINT ## 
                    start_evaluate = time.clock()
                    test_accuracy = evaluate()
                    saver = tf.train.Saver(tf.global_variables())
                    extra_time = extra_time + time.clock() - start_evaluate
                    print("--- %s CPU seconds ---" % (time.clock() - start_time - extra_time))
                    ckpt_file=os.path.join(FLAGS.data_dir, model_directory + "/drawmodel_" + str(i) + ".ckpt")
                    print("Model saved in file: %s" % saver.save(sess,ckpt_file))

    sess.close()
