# -*- coding: utf-8 -*-
"""
Generate computational graph to find the
relationship between Emission and weather data

CNN + Vanilla RNN
Series Connection

              -----
              |   |
              V   |
    CNN ---> RNN -- ---> Destination


But Now it Only have the CNN part - VGG19

Atmosphere Evolution under !Markov Chain!

CNN : Extract the characteristic of the original data
RNN : Evolve the data along chain

d(Emission_i+1)/dt = f(C_i, U_i, V_i, T_i, Q_i, time)

        DATA_i-1------------|
                            |------>Result_i
        DATA_i--------------|

ELIMINATE the time from the data
Convert TIME to CHAIN

Goal:
-------
U, V, Temp, Q, Concentrate, precursors ------>  Emission
        (100*100*n)        ------> (100*100)
-------

PAY ATTENTION :
------
+View the dimension of every tensor
+DO NOT use the same name for different variable

Version info :
------
+python > 3.5
+tensorflow > 1.00

07/2018 @ gatech MoSE 3229
09/2018 @ USTC ESS 1233
Fanghe : zfh1997 at mail.ustc.edu.cn(fzhao97 at gamil.com)
USTC-AEMOL
"""
import tensorflow as tf
import numpy as np
import time
from functools import reduce

def _message_display(string):
    """
    This function is only for display step message
    """
    print("========================================")
    print(string)
    print(time.asctime(time.localtime(time.time())))
    print("========================================")

def _weight_variable(shape, name):
    """Create a normal distribution, mu = 0, stderr = 0.1, min = -1, max = 1
    This should be only used in weight init
    """
    initial_weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_weight, name = name)

def _bias_variable(shape):
    """Create a bias matrix with constant initial 0.1
    """
    initial_bias = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_bias)

def _conv2d(x, w):
    """conv every step to 1 and same, out of the edge is auto to 0, then mutiply themselves
    Args:
    ------
    x : Input n-d array
        [batch, height, width, channels]
    w : A Tensor for filter(kernel)
        [filter_height, filter_width, in_channels, out_channels]

    Returns:
    ------
    tf.nn._conv2d : [batch, out_height, out_width, out_channels]
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def _maxpooling(input_data, strides):
    h_pool = tf.nn.max_pool(input_data, ksize = [1, 2, 2, 1], strides = strides, padding='SAME')
    return h_pool

def _full_connection(input_data, name, output_size):
    """An ensemble cell for FC layer

    Args:
    ------
    name : must be a string
    """
    product_dual = lambda x, y : x * y
    h_pool_sizes = int(reduce(product_dual, input_data.get_shape()))
    print(h_pool_sizes)

    w_fc1 = _weight_variable([h_pool_sizes, output_size], name = "w_fc" + name)
    b_fc1 = _bias_variable([output_size])
    h_pool_flat = tf.reshape(input_data, [-1, h_pool_sizes])
    h_fc = tf.nn.selu(tf.matmul(h_pool_flat, w_fc1) + b_fc1)
    print(name + str(h_fc.get_shape()))

    return h_fc

def _batch_normalization(input_data, out_size, axes):
    """An ensemble cell for BN layer

    Args:
    ------
    name : must be a string
    """
    mean, variance = tf.nn.moments(input_data, axes)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.0001
    BN_output = tf.nn.batch_normalization(input_data, mean, variance, shift, scale, variance_epsilon = epsilon) 

    return BN_output

class placeholder(object):
    """
    A class which ensemble all the placeholder
    """
    def __init__(self, shape):
        self.label = tf.placeholder(tf.float32, shape = (shape[0], shape[1], shape[2], 1))
        self.input = tf.placeholder(tf.float32, shape = shape)
        eigen_num = 288 #eigen_num is depend on the eigen of data itself
        self.eigen_label = tf.placeholder(tf.float32, shape = (eigen_num, shape[1], shape[2], 1)) 

class cnn_cell(object):
    """
    CNN Structure:

    VGG19

    Pay attention:
    ------
    After doing this function ALL var have been initialized
    If you change the shape, all the size of paras should also be changed
    In this Network , SeLu is somehow superior than ReLu
    """
    def __init__(self, shape, nn_input):
        #Retrive the shape data
        self.output = None
        self.shape = shape
        self.input = nn_input

    def _conv_cell(self, input_data, name, k_size):
        """An ensemble cell for conv_network

        Args:
        ------
        name : must be a string
        """
        w_conv2 = _weight_variable(k_size, name = "w_conv"+name)
        b_conv2 = _bias_variable([k_size[3]])
        h_conv = tf.nn.selu(_conv2d(input_data, w_conv2)+b_conv2)
        print(name + str(h_conv.get_shape()))
        return h_conv

    def graph_gen(self):
        data_len = self.shape[1]
        data_height = self.shape[2]
        data_channel = self.shape[3]
        data_batch = self.shape[0]
        data_num = data_channel * data_len * data_height
        #Dropout rate
        keep_prob = 0.8
        #Define placeholder
        label = tf.placeholder(tf.float32, shape = (data_batch, data_len, data_height, 1))

        _message_display("          CNN START           ")

        h_pool = self._conv_cell(self.input, "1_1", [3,3,data_channel,64])
        h_pool = self._conv_cell(h_pool, "1_1", [3,3,64,64])
        h_pool = _maxpooling(h_pool, [1,2,2,1])

        h_pool = self._conv_cell(h_pool, "2_1", [3,3,64,128])
        h_pool = self._conv_cell(h_pool, "2_2", [3,3,128,128])
        h_pool = _maxpooling(h_pool, [1,2,2,1])

        h_pool = self._conv_cell(h_pool, "3_1", [3,3,128,256])
        h_pool = self._conv_cell(h_pool, "3_2", [3,3,256,256])
        h_pool = self._conv_cell(h_pool, "3_3", [3,3,256,256])
        h_pool = self._conv_cell(h_pool, "3_4", [3,3,256,256])
        h_pool = _maxpooling(h_pool, [1,2,2,1])

        h_pool = self._conv_cell(h_pool, "3_1", [3,3,256,512])
        h_pool = self._conv_cell(h_pool, "3_2", [3,3,512,512])
        h_pool = self._conv_cell(h_pool, "3_3", [3,3,512,512])
        h_pool = self._conv_cell(h_pool, "3_4", [3,3,512,512])
        h_pool = _maxpooling(h_pool, [1,2,2,1])

        h_pool = self._conv_cell(h_pool, "4_1", [3,3,512,512])
        h_pool = self._conv_cell(h_pool, "4_2", [3,3,512,512])
        h_pool = self._conv_cell(h_pool, "4_3", [3,3,512,512])
        h_pool = self._conv_cell(h_pool, "4_4", [3,3,512,512])

        h_pool_drop = tf.nn.dropout(h_pool, keep_prob)
        h_fc = _full_connection(h_pool_drop, "1", 1024)
        h_fc = _batch_normalization(h_fc, 1024, [0, 1])
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
        h_fc = _full_connection(h_fc_drop, "2", 288)
        h_fc = _batch_normalization(h_fc, 288, [0, 1])
        cnn_output = tf.nn.softmax(h_fc)

        self.output = cnn_output
        _message_display("          CNN End           ")
    

class rnn_cell(object):

    def __init__(self, rnn_input):
        """
         HISTORICAL VERSION

        RNN creator
        This creator just create Vanilla RNN
        Use Dynamic_rnn, and it would loop according to the size of the input
            DATA_i-1 ----          ------>   RNN ------> Result   V
                        |          |          |                   |
                        |          |   vector & Weight_i-1        |
                        |          |          |                   |
            DATA_i   ------> CNN --------->  RNN ------> Result   V
                        |          |          |                   |
                        |          |   vector & Weight_i          |
                        |          |          |                   |
            DATA_i+1 ----          ------>   RNN ------> Result   V

        Inner Structure of RNN:
        As for RNN ,we use Vanilla RNN for atmosphere status make 
        evolution under !Markov Chain! Do not need long last memory.(LSTM)
        We can also set an offset for the input data

        #To use this:
            rnn = rnn_cell(cnn_output)
            rnn_output = rnn.output
        """
        # Inputs and outputs : [batch_size, max_time, cell_state_size]
        #Retrieve Data
        input_shape = rnn_input.get_shape()
        batch_size = input_shape[0]
        hidden_size = input_shape[1]
        rnn_input = tf.expand_dims(rnn_input, axis = 1)
        print("RNN input weight" + str(rnn_input.get_shape()))
        
        _message_display("          RNN START           ")

        #Cell Definition Part
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

        self.output, self.state = tf.nn.dynamic_rnn(rnn_cell, rnn_input, initial_state=initial_state, dtype=tf.float32)
        self.output = tf.reduce_sum(self.output, 1)
        print("RNN Cell Output : " + str(self.output.get_shape()))

        _message_display("          RNN END           ")
        

def output_summary_cell(rnn_output, eigen_label):
    """
    Process the output from the rnn_output

    Method : 
    Sum them up with label-eigen-matrix to make a final output
    which can be compared with label

    Return:
    ------
    output : a image with [1 * len * height * 1]
    """
    _message_display("          Output Process START           ")
    eigen_num = 288
    output_sum = np.zeros([100, 100])
    for i in range(eigen_num):
        output_sum = output_sum + rnn_output[0, i] * eigen_label[i, :, :, 0]

    # Make output dim as (1, 100, 100, 1)
    output_sum = tf.expand_dims(output_sum, 0)
    output_sum = tf.expand_dims(output_sum, 3)
    _message_display("          Output Process END           ")

    return output_sum


def graphgen(shape):
    """Generate a Graph with flexible loss
    The Structure of the graph:

        DATA_i-1 ----          -----------> Result  V
                    |          |                    |
                    |          |                    |
                    |          |                    |
        DATA_i   ------> CNN -------------> Result  V
                    |          |                    |
                    |          |                    |
                    |          |                    |
        DATA_i+1 ----          -----------> Result  V
    Args:
    ------
    shape : A tuple with 4 element

    Reutrns:
    ------
    init_op : A pointer which link to the graph
    nn_input, label : two placeholders for feed
    """
    _message_display("      Graph start       ")

    ph = placeholder(shape)
    label = ph.label
    nn_input = ph.input
    eigen_label = ph.eigen_label

    #Cell Connection Part
    cnn = cnn_cell(shape, nn_input)
    cnn.graph_gen()
    cnn_output = cnn.output
    nn_output = output_summary_cell(cnn_output, eigen_label)

    #+++Calculate Loss(euclidean)
    #This loss can predict the continuous var
    #2 ways to define the loss
    mean_bias = (tf.square(nn_output) - tf.square(label))[0, :, :, 0] / (100 * 100)
    _, std_nn = tf.nn.moments(nn_output, axes = [0, 1, 2, 3])
    _, std_label = tf.nn.moments(label, axes = [0, 1, 2, 3])
    cov_nn, cov_nn_op = tf.contrib.metrics.streaming_pearson_correlation(nn_output, label)

    loss = tf.norm(mean_bias, ord = np.inf)/(std_nn * std_label)

    acc, acc_op = tf.metrics.accuracy(label, nn_output)

    #To choose to use whether Adam or Grad_reduce is caused by Actual loss
    # Avail : GradientDescentOptimizer AdamOptimizer
    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

    #Initialized all variables and write the summary
    tf.summary.scalar('loss', loss)
    local_init_op = tf.local_variables_initializer()
    global_init_op = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    _message_display("      Graph End       ")
    
    dict_output = {
        "local_op" : local_init_op,\
        "global_op": global_init_op,\
        "train_op" : train_op,\
        "record" : merged,\
        "input" : nn_input,\
        "output" : nn_output,\
        "label" : label,\
        "eigen" : eigen_label,\
        "cov" : cov_nn,\
        "cov_op" : cov_nn_op}

    #return in form of dict
    return dict_output

if __name__  == "__main__":
    shape = (1,100,100,5)
    graphgen(shape)
