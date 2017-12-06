from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque


class TensorLSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self, num_units, num_lags, rank_vals, forget_bias=1.0, state_is_tuple=True, activation=tanh, reuse=None):
        super(TensorLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_lags = num_lags
        self._rank_vals = rank_vals
        self._forget_bias = forget_bias
        self._state_is_tuple= state_is_tuple
        self._activation = activation
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)
    
    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states):
        """Now we have multiple states, state->states"""
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            hs = ()
            for state in states:
            # every state is a tuple of (c,h)
                c, h = state
                hs += (h,)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
                hs += (h,)

        output_size = 4 * self._num_units
        concat = tensor_network_tt_einsum(inputs, hs, output_size, self._rank_vals, True)
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state
    
    

def _linear(args, output_size, bias, bias_start=0.0):
    total_arg_size = 0
    shapes= [a.get_shape() for a in args]
    for shape in shapes:
        total_arg_size += shape[1].value
    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable("weights", [total_arg_size, output_size], dtype=dtype)
        """y = [batch_size x total_arg_size] * [total_arg_size x output_size]"""
        res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            biases = vs.get_variable("biases", [output_size], dtype=dtype)
    return  nn_ops.bias_add(res,biases)

def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]

def _outer_product(batch_size, tensor, vector):
    """tensor-vector outer-product"""
    tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
    vector_flat = tf.expand_dims(vector, 1)
    res = tf.matmul(tensor_flat, vector_flat)
    new_shape =  [batch_size]+_shape_value(tensor)[1:]+_shape_value(vector)[1:]
    res = tf.reshape(res, new_shape )
    return res


def tensor_train_contraction(states_tensor, cores):
    # print("input:", states_tensor.name, states_tensor.get_shape().as_list())
    # print("mat_dims", mat_dims)
    # print("mat_ranks", mat_ranks)
    # print("mat_ps", mat_ps)
    # print("mat_size", mat_size)

    abc = "abcdefgh"
    ijk = "ijklmnopqrstuvwxy"

    def _get_indices(r):
        indices = "%s%s%s" % (abc[r], ijk[r], abc[r+1])
        return indices

    def _get_einsum(i, s2):
        #
        s1 = _get_indices(i)
        _s1 = s1.replace(s1[1], "")
        _s2 = s2.replace(s2[1], "")
        _s3 = _s2 + _s1
        _s3 = _s3[:-3] + _s3[-1:]
        s3 = s1 + "," + s2 + "->" + _s3
        return s3, _s3

    num_orders = len(cores)
    # first factor
    x = "z" + ijk[:num_orders] # "z" is the batch dimension
    
    # print(mat_core.get_shape().as_list())

    _s3 = x[:1] + x[2:] + "ab"
    einsum = "aib," + x + "->" + _s3
    x = _s3

    # print("einsum", einsum, cores[0].get_shape().as_list, states_tensor.get_shape().as_list)

    out_h = tf.einsum(einsum, cores[0], states_tensor)
    # print(out_h.name, out_h.get_shape().as_list())

    # 2nd - penultimate latent factor
    for i in range(1, num_orders):

        # We now compute the tensor inner product W * H, where W is decomposed
        # into a tensor-train with D factors A^i. Each factor A^i is a 3-tensor,
        # with dimensions [mat_rank[i], hidden_size, mat_rank[i+1] ]
        # The lag index, indexing the components of the state vector H,
        # runs from 1 <= i < K.

        # print mat_core.get_shape().as_list()

        einsum, x = ss, _s3 = _get_einsum(i, x)

        # print "order", i, ss

        out_h = tf.einsum(einsum, cores[i], out_h)
        # print(out_h.name, out_h.get_shape().as_list())

    # print "Squeeze out the dimension-1 dummy dim (first dim of 1st latent factor)"
    out_h = tf.squeeze(out_h, [1])
    return out_h


def tensor_network_tt_einsum(inputs, states, output_size, rank_vals, bias, bias_start=0.0):

    # print("Using Einsum Tensor-Train decomposition.")

    """tensor train decomposition for the full tenosr """
    num_orders = len(rank_vals)+1#alpha_1 to alpha_{K-1}
    num_lags = len(states)
    batch_size = tf.shape(inputs)[0] 
    state_size = states[0].get_shape()[1].value #hidden layer size
    input_size= inputs.get_shape()[1].value
    total_state_size = (state_size * num_lags + 1 )

    # These bookkeeping variables hold the dimension information that we'll
    # use to store and access the transition tensor W efficiently.
    mat_dims = np.ones((num_orders,)) * total_state_size

    # The latent dimensions used in our tensor-train decomposition.
    # Each factor A^i is a 3-tensor, with dimensions [a_i, hidden_size, a_{i+1}]
    # with dimensions [mat_rank[i], hidden_size, mat_rank[i+1] ]
    # The last
    # entry is the output dimension, output_size: that dimension will be the
    # output.
    mat_ranks = np.concatenate(([1], rank_vals, [output_size]))

    # This stores the boundary indices for the factors A. Starting from 0,
    # each index i is computed by adding the number of weights in the i'th
    # factor A^i.
    mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * mat_dims * mat_ranks[1:])),dtype=np.int32)
    mat_size = mat_ps[-1]

    # Compute U * x
    weights_x = vs.get_variable("weights_x", [input_size, output_size] )
    out_x = tf.matmul(inputs, weights_x)

    # Get a variable that holds all the weights of the factors A^i of the
    # transition tensor W. All weights are stored serially, so we need to do
    # some bookkeeping to keep track of where each factor is stored.
    mat = vs.get_variable("weights_h", mat_size) # h_z x h_z... x output_size

    #mat = tf.Variable(mat, name="weights")
    states_vector = tf.concat(states, 1)
    states_vector = tf.concat( [states_vector, tf.ones([batch_size, 1])], 1)
    """form high order state tensor"""
    states_tensor = states_vector
    for order in range(num_orders-1):
        states_tensor = _outer_product(batch_size, states_tensor, states_vector)

    # print("tensor product", states_tensor.name, states_tensor.get_shape().as_list())
    cores = []
    for i in range(num_orders):
        # Fetch the weights of factor A^i from our big serialized variable weights_h.
        mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
        mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])   
        cores.append(mat_core)
        
    out_h = tensor_train_contraction(states_tensor, cores)
    # Compute h_t = U*x_t + W*H_{t-1}
    res = tf.add(out_x, out_h)

    # print "END OF CELL CONSTRUCTION"
    # print "========================"
    # print ""

    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])

    return nn_ops.bias_add(res,biases)
