# -*- coding: utf-8 -*-
"""
Main Program to use the wrf data to derive emission distribution

                  -----
                  |   |
                  V   |
    CNN ---> Vanilla RNN -- ---> Destination

Module load :
------
+graph_gen.py
+init_data.py

Version info :
------
+python > 3.5

09/2018 @ USTC ESS 1233
Fanghe : zfh1997 at mail.ustc.edu.cn(fzhao97 at gamil.com)
USTC-AEMOL
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
#user module
import graph_gen
import wrf_out_input

def _message_display(string):
    """
    This function is only for display step message
    """
    print("========================================")
    print(string)
    print(time.asctime(time.localtime(time.time())))
    print("========================================")

if __name__ == '__main__':
    debug = True
    wrf_data, wrf_month, wrf_hour = wrf_out_input.read_wrf_data()
    emis_label, emis_eigen = wrf_out_input.emis_cohere(wrf_month, wrf_hour)

    train_number = 1500
    wrf_data = wrf_data[0:train_number, :, :, :]
    emis_data = emis_label[0:train_number, :, :, :]

    work_path = r"D:\AEMOL\Tensorflow\MIX"
    os.chdir(work_path)
    print(str(wrf_data.shape) + "  " +str(emis_label.shape))
    wrf_shape = wrf_data.shape
    shape = (1, wrf_shape[1], wrf_shape[2], wrf_shape[3])
    print(shape)

    #train_op, merged, nn_input, label, eigen_label = graph_gen_cnn_only.graphgen(shape)
    dict_graph = graph_gen.graphgen(shape)
    local_op = dict_graph['local_op']
    global_op = dict_graph['global_op']
    train_op = dict_graph['train_op']
    merged = dict_graph['record']
    nn_input = dict_graph['input']
    nn_output = dict_graph['output']
    label = dict_graph['label']
    eigen_label = dict_graph['eigen']
    cov = dict_graph['cov']
    cov_op = dict_graph['cov_op']

    ##Initialize all Variables, and init_op is the point of the whole graph
    #sess = tf.InteractiveSession()
    _message_display("         Train Process Start            ")
    
    output_list = []
    train_batch = wrf_shape[0]
    with tf.Session() as sess:
        with tf.summary.FileWriter('./graphs', sess.graph) as writer:
            for i in range(train_batch):
                print(i)
                #To Retrieve single data from batch
                image_feed = wrf_data[i, :, :, :][np.newaxis, :, :, :]
                label_feed = emis_data[i, :, :, :][np.newaxis, :, :, :]
                # Run a Actual Session
                sess.run(local_op, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                sess.run(global_op, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                sess.run(train_op, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                output_list.append(sess.run(nn_output, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen}))
                #total_output = sess.run(nn_output, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                summary = sess.run(merged, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                #sess.run(acc_op, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                #acc_out = sess.run([acc], feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                sess.run(cov_op, feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                cov_out = sess.run([cov], feed_dict={nn_input:image_feed, label:label_feed, eigen_label:emis_eigen})
                print(cov_out)
                writer.add_summary(summary)

    _message_display("         Train Process Done             ")
    np.save("output", np.array(output_list))

    _message_display("         Graphic Process Start          ")
    #plt.subplot(311)
    #plt.imshow(output_list[10][0, :, :, 0])
    #plt.subplot(312)
    #plt.imshow(emis_data[10, :, :, 0])
    #plt.subplot(313)
    #plt.imshow(output_list[10][0, :, :, 0] - emis_data[10, :, :, 0])
    #plt.show()
    _message_display("         Graphic Process Done          ")
