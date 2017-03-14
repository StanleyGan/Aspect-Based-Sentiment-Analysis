#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:03:15 2017

@author: stanleygan
"""
import tensorflow as tf
from data import data
import random 

if __name__ == "__main__":
    d = data("/home/stanleygan/Documents/Deep_Learning/project/customer_review_data/")
    train_feat, train_lab, val_feat, val_lab, test_feat, test_lab = d.getData()
    
    vocab = d.getVocab()
    sizeVocab = len(vocab)
#    num_hidden = 24
#    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    data = tf.placeholder(tf.float32, [None, sizeVocab])
    target = tf.placeholder(tf.float32, [None, sizeVocab])
    keep_prob = tf.placeholder(tf.float32)
    
#    value, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    W1 = tf.Variable(tf.random_uniform(shape=(sizeVocab, 250), minval=-0.01, maxval=0.01))
    b1 = tf.Variable(tf.constant(0.1, shape=[250]))
    
    h = tf.nn.relu(tf.matmul(data, W1) + b1)
    h_dropout = tf.nn.dropout(h, keep_prob)
    
    W2 = tf.Variable(tf.random_uniform((250,sizeVocab), minval=-0.01, maxval=0.01))
    b2 = tf.Variable(tf.constant(0.1, shape=[500]))
    
    pred = tf.matmul(h_dropout, W2) + b2
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, target))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        random_index = range(train_feat.shape[0])
        random.shuffle(random_index)
    	
        batch_track = 0
        batch_size = 5

        for i in range(10000):
            batch_ind = random_index[batch_track : batch_track + batch_size]
            feat_batch = [train_feat[j] for j in batch_ind]
            lab_batch = [train_lab[k] for k in batch_ind]
            
            [_, loss_val] = sess.run([train_step, cross_entropy], feed_dict={data: feat_batch, target: lab_batch, keep_prob: 0.5})

            if i%5 == 0:
                print("step %d, loss %g"%(i, loss_val))
            
            if batch_track >= 282:
                random.shuffle(random_index)
                batch_track = 0
            else:
                batch_track = (batch_track + batch_size) % 282
        
        #validate data
        prediction = tf.nn.softmax(pred)
        prediction = prediction.eval(session=sess, feed_dict={data:val_feat, keep_prob:1.0})