"""Functions for downloading and reading time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed

# start of sequences
SOS = 0

def slide_window(a, window):
    """ Extract examples from time series"""
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    examples = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    inp = examples[:-1]
    out = examples[1:]
    return inp, out


def normalize_columns(arr):
    def _norm_col(arr):
        stats = np.zeros((2,))
        stats[0] = arr.min()
        stats[1] = arr.max()
        if abs(stats[0] - stats[1]) < 1e-10:
            pass 
        else:
            arr = (arr - arr.min() ) / (arr.max() - arr.min()) 
        return arr, stats
    # Normalize each feature dimension
    n_dim = arr.shape[-1]
    stats = np.zeros((2,n_dim))
    if np.ndim(arr) ==2:
        for d in range(n_dim):
            arr[:,d], stats[:,d]= _norm_col(arr[:,d])
    elif np.ndim(arr)==3:
        for d in range(n_dim):
            arr[:,:,d], stats[:,d] = _norm_col(arr[:,:,d])  
    return arr, stats

def denormalize_colums(arr, stats):
    def _denorm_col(arr, stats):
        arr  = arr  * (stats[1]- stats[0]) +  stats[0]
        return arr
    
    n_dim = arr.shape[-1]
    for d in range(n_dim):
        arr[:,:,d]  = _denorm_col(arr[:,:,d], stats[:,d])
    return arr
                    
class DataSet(object):

    def __init__(self,
                     data,
                     num_steps,
                     seed=None):
        """Construct a DataSet.
        Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
     
        inps, outs = slide_window(data, num_steps)
        # inps = data[:,:num_steps,:]
        # outs = data[:,1:num_steps+1,:]

        assert inps.shape[0] == outs.shape[0], (
                'inps.shape: %s outs.shape: %s' % (inps.shape, outs.shape))


        self._num_examples = inps.shape[0]
        self._inps = inps
        self._outs = outs
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def inps(self):
        return self._inps

    @property
    def outs(self):
        return self._outs

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._inps = self.inps[perm0]
            self._outs = self.outs[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            inps_rest_part = self._inps[start:self._num_examples]
            outs_rest_part = self._outs[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._inps = self.inps[perm]
                self._outs = self.outs[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            inps_new_part = self._inps[start:end]
            outs_new_part = self._outs[start:end]
            return np.concatenate((inps_rest_part, inps_new_part), axis=0) , np.concatenate((outs_rest_part, outs_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._inps[start:end], self._outs[start:end]

class DataSetS2S(object):
    def __init__(self,
                     data,
                     num_steps,
                     num_test_steps=None,
                     seed=None):
        """Construct a DataSet.
        Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
     
        #inps, outs = slide_window(data, num_steps)
        #inps = data[:,:num_steps,:]
        #outs = data[:,1:num_steps+1,:]
        
        time_len = data.shape[1]
        if num_test_steps is None:
            num_test_steps=  time_len-num_steps 
        enc_inps = data[:,:num_steps, :]
        dec_inps = np.insert(data[:,num_steps:num_steps+num_test_steps-1,:], 0, SOS, axis=1)
        #dec_outs = np.insert(data[:,num_steps:num_steps+num_test_steps,:], num_test_steps, EOS, axis=1)
        dec_outs = data[:,num_steps:num_steps+num_test_steps,:]

        assert enc_inps.shape[0] == dec_outs.shape[0], (
                'inps.shape: %s outs.shape: %s' % (inps.shape, outs.shape))


        self._num_examples = enc_inps.shape[0]
        self._enc_inps = enc_inps
        self._dec_inps = dec_inps
        self._dec_outs = dec_outs
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def enc_inps(self):
        return self._enc_inps
    @property
    def dec_inps(self):
        return self._dec_inps
    @property
    def dec_outs(self):
        return self._dec_outs

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._enc_inps = self.enc_inps[perm0]
            self._dec_inps = self.dec_inps[perm0]
            self._dec_outs = self.dec_outs[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            enc_inps_rest_part = self._enc_inps[start:self._num_examples]
            dec_inps_rest_part = self._dec_inps[start:self._num_examples]
            dec_outs_rest_part = self._dec_outs[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._enc_inps = self.enc_inps[perm]
                self._dec_inps = self.dec_inps[perm]
                self._dec_outs = self.dec_outs[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            enc_inps_new_part = self._enc_inps[start:end]
            dec_inps_new_part = self._dec_inps[start:end]
            dec_outs_new_part = self._dec_outs[start:end]
            return np.concatenate((enc_inps_rest_part, enc_inps_new_part), axis=0), \
                   np.concatenate((dec_inps_rest_part, dec_inps_new_part), axis=0), \
                   np.concatenate((dec_outs_rest_part, dec_outs_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._enc_inps[start:end], self._dec_inps[start:end], self._dec_outs[start:end]
        
def read_data_sets(data_path, s2s, n_steps,
                                n_test_steps = None,
                                val_size = 0.1, 
                                test_size = 0.1, 
                                seed=None):
    print("loading time series ...")
    data = np.load(data_path)
    # Expand the dimension if univariate time series
    if (np.ndim(data)==1):
            data = np.expand_dims(data, axis=1)
    print("input type ",type( data), np.shape(data))

    # Normalize the data
    print("normalize to (0-1)")
    data, _ = normalize_columns(data)
    ntest = int(round(len(data) * (1.0 - test_size)))
    nval = int(round(len(data[:ntest]) * (1.0 - val_size)))

    train_data, valid_data, test_data = data[:nval, ], data[nval:ntest, ], data[ntest:,]

    train_options = dict(num_steps=n_steps, num_test_steps=n_test_steps, seed=seed)
    if s2s == True:
        train = DataSetS2S(train_data, **train_options)
        valid = DataSetS2S(valid_data, **train_options)
        test = DataSetS2S(test_data, **train_options)
    else:
        train = DataSet(train_data, **train_options)
        valid = DataSet(valid_data, **train_options)
        test = DataSet(test_data, **train_options)     

    stats ={}
    stats['num_examples'] = data.shape[0]
    stats['num_steps'] = data.shape[1]
    stats['num_input'] = data.shape[-1]

    return base.Datasets(train=train, validation=valid, test=test), stats

