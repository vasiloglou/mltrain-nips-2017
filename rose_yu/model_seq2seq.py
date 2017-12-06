
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from trnn import *
from trnn_imply import *


def TLSTM(enc_inps, dec_inps, is_training, config):
    def tlstm_cell():
        return TensorLSTMCell(config.hidden_size, config.num_lags, config.rank_vals)
    print('Training -->') if is_training else print('Testing -->')
    cell= tlstm_cell() 
    #if is_training and config.keep_prob < 1:
    #    cell = tf.contrib.rnn.DropoutWrapper(
    #      cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        print(' '*10+'Create Encoder ...')
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)
    with tf.variable_scope("Decoder", reuse=None):
        print(' '*10+'Create Decoder ...')
        config.inp_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs   


def RNN(enc_inps, dec_inps,is_training, config):
    def rnn_cell():
        return tf.contrib.rnn.BasicRNNCell(config.hidden_size)
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          rnn_cell(), output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [rnn_cell() for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.inp_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    

def LSTM(enc_inps, dec_inps, is_training, config):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Define a lstm cell with tensorflow
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(config.hidden_size,forget_bias=1.0, reuse=None)
    #if is_training and config.keep_prob < 1:
    #    cell = tf.contrib.rnn.DropoutWrapper(
    #      lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(config.num_layers)])

    # Get encoder output
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell, enc_inps, True, config)
    # Get decoder output
    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states)
    
    return dec_outs


